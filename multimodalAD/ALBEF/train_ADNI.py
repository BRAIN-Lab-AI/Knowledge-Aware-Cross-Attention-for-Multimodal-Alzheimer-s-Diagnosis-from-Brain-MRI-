'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

import argparse
import os
from ruamel.yaml import YAML 
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import torchsummary

from models.model_pretrain3D import MultiModal3DClassifier

import utils
from dataset import create_dataset3d, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer

import warnings
from models.vit import interpolate_pos_embed 

warnings.filterwarnings(
    'ignore',
    message='Using TorchIO images without a torchio.SubjectsLoader in PyTorch >= 2.3 might have unexpected consequences'
)


def train(model, data_loader, optimizer, epoch, warmup_steps, device, scheduler, config):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_cls', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 10   
    step_size = 100
    warmup_iterations = warmup_steps*step_size  
    
    if args.distributed:
        data_loader.sampler.set_epoch(epoch)

    t = time.time()
    for i, (mri, pet, label) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        t = time.time()        
        optimizer.zero_grad()
  
        mri = mri.to(device,non_blocking=True) 
        pet = pet.to(device,non_blocking=True)
        label = label.long().to(device,non_blocking=True)

        if epoch>0:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader)) 
        
        # NOTE: train=True is implicitly used here
        loss_cls, loss_mlm, loss_ita, loss_itm, _ = model(mri, pet, label, alpha)
        loss = loss_cls + loss_mlm + loss_ita + loss_itm
          
        loss.backward()
        optimizer.step()
        
        metric_logger.update(loss_cls=loss_cls.item())
        metric_logger.update(loss_mlm=loss_mlm.item())
        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size)

        t = time.time()
        
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

def validata_and_test(model, data_loader, epoch, device, config, val=True):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss_cls', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('real_acc', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('acc', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Valid Epoch: [{}]'.format(epoch) if val else 'Test Epoch: [{}]'.format(epoch)
    with torch.no_grad():
        for i, (mri, pet, label) in enumerate(metric_logger.log_every(data_loader, 10, header)):
            mri = mri.to(device,non_blocking=True) 
            pet = pet.to(device,non_blocking=True)
            label = label.long().to(device,non_blocking=True)
            loss_cls, loss_mlm, loss_ita, loss_itm, output = model(mri, pet, label, config['alpha'], train=False)
            metric_logger.update(loss_mlm=loss_mlm.item())
            metric_logger.update(loss_ita=loss_ita.item())
            metric_logger.update(loss_itm=loss_itm.item())
            metric_logger.update(loss_cls=loss_cls.item())
            metric_logger.update(real_acc=(torch.argmax(output, dim=1) == label).float().mean().item())
            
            if hasattr(model, 'module'):
                metrics = model.module.get_metrics(output, label)
            else:
                metrics = model.get_metrics(output, label)
            
            metric_logger.update(**metrics)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


# --- NEW FEATURE EXTRACTION FUNCTION ---
def extract_features(model, data_loader, device):
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for i, (mri, pet, label) in enumerate(data_loader):
            mri = mri.to(device)
            pet = pet.to(device)
            label = label.long().cpu().numpy()
            
            # The MultiModal3DClassifier model returns the fused embedding 
            # when its base ALBEF3D model's forward method is called 
            # with train=False and no label.
            # However, since the classifier is the head, we need to adapt the model call 
            # to return the layer right before the classification head.
            
            # For simplicity and robust extraction: We access the last hidden state 
            # (which contains the fused feature vector for the CLS token) directly 
            # from the fusion encoder inside the model.

            # Access the base model (unwrapped DDP or raw model)
            base_model = model.module if hasattr(model, 'module') else model

            # Get the input embeddings for the fusion encoder
            image_embeds = base_model.visual_encoder(mri) 
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
            pet_embeds = base_model.pet_encoder(pet)
            pet_atts = torch.ones(pet_embeds.size()[:-1], dtype=torch.long).to(device)

            # Run through the fusion encoder
            fusion_output = base_model.fusion_encoder(
                encoder_embeds=pet_embeds, 
                attention_mask=pet_atts,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,      
                return_dict=True,
                mode='multimodal'
            )
            
            # Extract the CLS token feature (index 0) from the last layer
            fused_feature = fusion_output.last_hidden_state[:, 0, :]
            
            all_features.append(fused_feature.cpu().numpy())
            all_labels.append(label)
            
    # Concatenate all batches
    features_array = np.concatenate(all_features, axis=0)
    labels_array = np.concatenate(all_labels, axis=0)
    
    return features_array, labels_array


def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']    

    #### Dataset #### 
    print("Creating dataset")
    t = time.time()
    datasets = [create_dataset3d('adni_cls_train', config), create_dataset3d('adni_cls_val', config), create_dataset3d('adni_cls_test', config)]
    print('Data loading time:', time.time()-t)
    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True, False, False], num_tasks, global_rank)         
    else:
        samplers = [None, None, None]

    tr_data_loader, val_data_loader, test_data_loader = create_loader(
        datasets,
        samplers,
        # Use batch size 1 for feature extraction to ensure 1-to-1 patient feature mapping, 
        # but keep it at config['batch_size'] for training for speed.
        batch_size=[config['batch_size'], config['batch_size'], config['batch_size']], 
        num_workers=[0, 0, 0], 
        is_trains=[True, False, False], 
        collate_fns=[None, None, None])

    # Feature extraction loaders (using batch_size 1 for guaranteed feature per patient)
    tr_feat_loader, val_feat_loader, test_feat_loader = create_loader(
        datasets,
        [None, None, None],
        batch_size=[1, 1, 1], 
        num_workers=[0, 0, 0], 
        is_trains=[False, False, False], 
        collate_fns=[None, None, None])


    #### Model #### 
    print("Creating model")
    model = MultiModal3DClassifier(patch_size=config['patch_size'], config=config, class_num=3)
    
    model = model.to(device)
    
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)  

    
    assert args.checkpoint != '', 'Please specify the checkpoint to load'

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False) 
    state_dict = checkpoint['model']
    
    if 'visual_encoder.pos_embed' in state_dict:
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
    
    if 'visual_encoder_m.pos_embed' in state_dict:
        m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'], model.visual_encoder_m)
        state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped
        
    if 'pet_encoder.pos_embed' in state_dict:
        pet_pos_embed_reshaped = interpolate_pos_embed(state_dict['pet_encoder.pos_embed'], model.pet_encoder)
        state_dict['pet_encoder.pos_embed'] = pet_pos_embed_reshaped
        
    if 'pet_encoder_m.pos_embed' in state_dict:
        pet_m_pos_embed_reshaped = interpolate_pos_embed(state_dict['pet_encoder_m.pos_embed'], model.pet_encoder_m)
        state_dict['pet_encoder_m.pos_embed'] = pet_m_pos_embed_reshaped

    model.load_state_dict(state_dict, strict=False)
    print('load checkpoint from %s'%args.checkpoint)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    print("Start training")
    start_time = time.time()
    best_val_loss_cls = float('inf')

    for epoch in range(start_epoch, max_epoch):
        
        if epoch>0:
            lr_scheduler.step(epoch+warmup_steps)  
            
        train_stats = train(model, tr_data_loader, optimizer, epoch, warmup_steps, device, lr_scheduler, config)
        val_stats = validata_and_test(model, val_data_loader, epoch, device, config, val=True)
        if utils.is_main_process():  
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                        }                     
            
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")

            log_stats = {**{f'val_{k}': v for k, v in val_stats.items()},
                         'epoch': epoch,
                        }
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")
            
            if float(val_stats['loss_cls']) < best_val_loss_cls:

                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
                torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch))

        if float(val_stats['loss_cls']) < best_val_loss_cls:
            best_val_loss_cls = float(val_stats['loss_cls'])
            test_stats = validata_and_test(model, test_feat_loader, epoch, device, config, val=False)
            if utils.is_main_process():
                log_stats = {**{f'test_{k}': v for k, v in test_stats.items()},
                             'epoch': epoch,
                            }
                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")

        if args.distributed:
            dist.barrier()  
                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 


    # --- FEATURE EXTRACTION AFTER TRAINING ---
    print("\n--- Starting Feature Extraction for TabPFN ---")
    
    # Train Features
    tr_features, tr_labels = extract_features(model, tr_feat_loader, device)
    np.savez_compressed(os.path.join(args.output_dir, 'tr_feat_fusioned.npz'), fusioned=tr_features)
    np.savez_compressed(os.path.join(args.output_dir, 'tr_label.npz'), label=tr_labels)

    # Val Features
    val_features, val_labels = extract_features(model, val_feat_loader, device)
    np.savez_compressed(os.path.join(args.output_dir, 'val_feat_fusioned.npz'), fusioned=val_features)
    np.savez_compressed(os.path.join(args.output_dir, 'val_label.npz'), label=val_labels)
    
    # Test Features
    test_features, test_labels = extract_features(model, test_feat_loader, device)
    np.savez_compressed(os.path.join(args.output_dir, 'test_feat_fusioned.npz'), fusioned=test_features)
    np.savez_compressed(os.path.join(args.output_dir, 'test_label.npz'), label=test_labels)

    print(f"Features saved successfully to {args.output_dir}")
    print(f"Train Features Shape: {tr_features.shape}")
    print(f"Test Features Shape: {test_features.shape}")

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Pretrain.yaml')
    parser.add_argument('--checkpoint', default='') 
    parser.add_argument('--output_dir', default='Pretrain/')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    # add timestep
    args.output_dir = args.output_dir + '_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    yaml = YAML()
    config = yaml.load(open(args.config, 'r'))

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)