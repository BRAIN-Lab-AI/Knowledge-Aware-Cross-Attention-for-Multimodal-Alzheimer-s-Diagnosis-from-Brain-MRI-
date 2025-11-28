
'''
Adapted from ALBEF for 3D Multimodal (MRI + PET) Data
'''
from functools import partial
from models.vit import VisionTransformer, interpolate_pos_embed
from models.xbert import BertConfig, BertModel

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn

import numpy as np

class PatchEmbed3D(nn.Module):
    """ 3D Image to Patch Embedding
    """
    def __init__(self, img_size=128, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        # Handle list or tuple inputs for resolution
        if isinstance(img_size, (list, tuple)):
            if len(img_size) == 3:
                img_size = tuple(img_size)
            else:
                # Fallback to cubic if format is unexpected
                img_size = (img_size[0], img_size[0], img_size[0])
        elif isinstance(img_size, int):
            img_size = (img_size, img_size, img_size)

        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
            
        num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x shape: [B, C, D, H, W]
        # output: [B, Embed, D', H', W'] -> flatten -> [B, Embed, N] -> transpose -> [B, N, Embed]
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class VisionTransformer3D(VisionTransformer):
    """ 3D Vision Transformer adapting timm's 2D ViT
    """
    def __init__(self, img_size=128, patch_size=16, in_chans=1, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None):
        
        # Initialize parent class (standard ViT from ALBEF/timm) 
        # We pass a dummy img_size=224 because we override the embedding layer immediately after
        super().__init__(img_size=224, patch_size=16, in_chans=in_chans, embed_dim=embed_dim, depth=depth,
                         num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                         representation_size=representation_size, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                         drop_path_rate=drop_path_rate, norm_layer=norm_layer)
        
        # Override patch embedding for 3D
        self.patch_embed = PatchEmbed3D(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Override position embedding for 3D patches (1 for CLS token + num_patches)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

class ALBEF3D(nn.Module):
    def __init__(self, config=None, patch_size=16):
        super().__init__()
        
        embed_dim = config['embed_dim']
        vision_width = config['vision_width']
        image_res = config['image_res']
        
        # MRI Encoder
        self.visual_encoder = VisionTransformer3D(
            img_size=image_res, patch_size=patch_size, embed_dim=vision_width, depth=config['depth'], 
            num_heads=config['num_heads'], mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
            
        # PET Encoder (Treating PET as the second modality)
        self.pet_encoder = VisionTransformer3D(
            img_size=image_res, patch_size=patch_size, embed_dim=vision_width, depth=config['depth'], 
            num_heads=config['num_heads'], mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

        # Fusion Encoder (Modified BERT)
        bert_config = BertConfig.from_json_file('./configs/bert_config.json')
        bert_config.hidden_size = vision_width 
        bert_config.num_hidden_layers = 6 # Multimodal layers
        self.fusion_encoder = BertModel(config=bert_config, add_pooling_layer=False)

        # Projection heads
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.pet_proj = nn.Linear(vision_width, embed_dim)

        self.itm_head = nn.Linear(vision_width, 2)
        self.mlm_head = nn.Linear(vision_width, vision_width) 

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])   
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']  

        # Momentum Encoders
        self.visual_encoder_m = VisionTransformer3D(
            img_size=image_res, patch_size=patch_size, embed_dim=vision_width, depth=config['depth'], 
            num_heads=config['num_heads'], mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.pet_encoder_m = VisionTransformer3D(
            img_size=image_res, patch_size=patch_size, embed_dim=vision_width, depth=config['depth'], 
            num_heads=config['num_heads'], mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
            
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.pet_proj_m = nn.Linear(vision_width, embed_dim)

        self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                            [self.pet_encoder,self.pet_encoder_m],
                            [self.vision_proj,self.vision_proj_m],
                            [self.pet_proj,self.pet_proj_m]]

        self.copy_params()

        # Queue
        self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("pet_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.pet_queue = nn.functional.normalize(self.pet_queue, dim=0)

    def forward(self, mri, pet, alpha=0):
        # MRI Features
        image_embeds = self.visual_encoder(mri) 
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(mri.device)
        image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]), dim=-1)

        # PET Features
        pet_embeds = self.pet_encoder(pet)
        pet_atts = torch.ones(pet_embeds.size()[:-1], dtype=torch.long).to(pet.device)
        pet_feat = F.normalize(self.pet_proj(pet_embeds[:,0,:]), dim=-1)

        # --- ITA Loss (Image-Text Alignment / MRI-PET Alignment) ---
        with torch.no_grad():
            self._momentum_update()
            image_embeds_m = self.visual_encoder_m(mri) 
            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:,0,:]), dim=-1)  
            image_feat_all = torch.cat([image_feat_m.t(),self.image_queue.clone().detach()],dim=1)                                         
            
            pet_embeds_m = self.pet_encoder_m(pet) 
            pet_feat_m = F.normalize(self.pet_proj_m(pet_embeds_m[:,0,:]), dim=-1) 
            pet_feat_all = torch.cat([pet_feat_m.t(),self.pet_queue.clone().detach()],dim=1)

            sim_i2p_m = image_feat_m @ pet_feat_all / self.temp 
            sim_p2i_m = pet_feat_m @ image_feat_all / self.temp 

            sim_targets = torch.zeros(sim_i2p_m.size()).to(mri.device)
            sim_targets.fill_diagonal_(1)
            
            sim_i2p_targets = alpha * F.softmax(sim_i2p_m, dim=1) + (1 - alpha) * sim_targets
            sim_p2i_targets = alpha * F.softmax(sim_p2i_m, dim=1) + (1 - alpha) * sim_targets        

        sim_i2p = image_feat @ pet_feat_all / self.temp 
        sim_p2i = pet_feat @ image_feat_all / self.temp 
                             
        loss_i2p = -torch.sum(F.log_softmax(sim_i2p, dim=1)*sim_i2p_targets,dim=1).mean()
        loss_p2i = -torch.sum(F.log_softmax(sim_p2i, dim=1)*sim_p2i_targets,dim=1).mean() 

        loss_ita = (loss_i2p + loss_p2i)/2

        self._dequeue_and_enqueue(image_feat_m, pet_feat_m)

        # --- ITM Loss (Image-Text Matching / MRI-PET Matching) ---
        output_pos = self.fusion_encoder(encoder_embeds=pet_embeds, 
                                         attention_mask=pet_atts,
                                         encoder_hidden_states=image_embeds,
                                         encoder_attention_mask=image_atts,      
                                         return_dict=True,
                                         mode='multimodal',
                                         )
        
        with torch.no_grad():
            bs = mri.size(0)
            weights_i2p = F.softmax(sim_i2p[:,:bs],dim=1)
            weights_p2i = F.softmax(sim_p2i[:,:bs],dim=1)
            weights_i2p.fill_diagonal_(0)
            weights_p2i.fill_diagonal_(0)

        # Select negatives
        image_embeds_neg = []    
        image_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_p2i[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
            image_atts_neg.append(image_atts[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg,dim=0)   
        image_atts_neg = torch.stack(image_atts_neg,dim=0)      

        pet_embeds_neg = []
        pet_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2p[b], 1).item()
            pet_embeds_neg.append(pet_embeds[neg_idx])
            pet_atts_neg.append(pet_atts[neg_idx])
        pet_embeds_neg = torch.stack(pet_embeds_neg,dim=0)   
        pet_atts_neg = torch.stack(pet_atts_neg,dim=0)

        pet_embeds_all = torch.cat([pet_embeds, pet_embeds_neg],dim=0)
        pet_atts_all = torch.cat([pet_atts, pet_atts_neg],dim=0)     

        image_embeds_all = torch.cat([image_embeds_neg, image_embeds],dim=0)
        image_atts_all = torch.cat([image_atts_neg, image_atts],dim=0)

        output_neg = self.fusion_encoder(encoder_embeds=pet_embeds_all, 
                                         attention_mask=pet_atts_all,
                                         encoder_hidden_states=image_embeds_all,
                                         encoder_attention_mask=image_atts_all,      
                                         return_dict=True,
                                         mode='multimodal',
                                         )

        vl_embeddings = torch.cat([output_pos.last_hidden_state[:,0,:], output_neg.last_hidden_state[:,0,:]], dim=0)
        vl_output = self.itm_head(vl_embeddings)            

        itm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)],dim=0).to(mri.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)     

        # --- MLM Loss (Masked Modeling) ---
        # Placeholder for MLM loss (simplified for now)
        loss_mlm = torch.tensor(0.0).to(mri.device) 
        
        return loss_mlm, loss_ita, loss_itm

    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  
                param_m.requires_grad = False  

    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, pet_feat):
        image_feats = concat_all_gather(image_feat)
        pet_feats = concat_all_gather(pet_feat)
        batch_size = image_feats.shape[0]
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0 
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.pet_queue[:, ptr:ptr + batch_size] = pet_feats.T
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

class MultiModal3DClassifier(ALBEF3D):
    def __init__(self, config=None, patch_size=16, class_num=3):
        super().__init__(config, patch_size)
        self.cls_head = nn.Linear(config['vision_width'], class_num)

    def forward(self, mri, pet, label=None, alpha=0, train=True):
        image_embeds = self.visual_encoder(mri) 
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(mri.device)
        pet_embeds = self.pet_encoder(pet)
        pet_atts = torch.ones(pet_embeds.size()[:-1], dtype=torch.long).to(pet.device)

        output = self.fusion_encoder(encoder_embeds=pet_embeds, 
                                     attention_mask=pet_atts,
                                     encoder_hidden_states=image_embeds,
                                     encoder_attention_mask=image_atts,      
                                     return_dict=True,
                                     mode='multimodal',
                                     )
        
        prediction = self.cls_head(output.last_hidden_state[:,0,:])
        
        loss_cls = F.cross_entropy(prediction, label) if label is not None else 0
        
        # Aux losses set to 0 for classification fine-tuning simplicity
        loss_mlm = torch.tensor(0.0).to(mri.device)
        loss_ita = torch.tensor(0.0).to(mri.device)
        loss_itm = torch.tensor(0.0).to(mri.device)
        
        return loss_cls, loss_mlm, loss_ita, loss_itm, prediction

    def get_metrics(self, output, label):
        preds = torch.argmax(output, dim=1)
        acc = (preds == label).float().mean()
        return {'acc': acc.item()}

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** FIX: Handles non-distributed (single GPU) mode safely ***
    """
    if not dist.is_initialized():
        return tensor

    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output