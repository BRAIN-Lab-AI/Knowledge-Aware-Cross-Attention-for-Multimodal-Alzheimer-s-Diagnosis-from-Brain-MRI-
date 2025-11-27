
import json
import os
import random

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import pre_caption
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split


class re_train_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.img_ids = {}   
        
        n = 0
        for ann in self.ann:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        image_path = os.path.join(self.image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        
        caption = pre_caption(ann['caption'], self.max_words) 

        return image, caption, self.img_ids[ann['image_id']]
    
    

class re_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):        
        self.ann = json.load(open(ann_file,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words 
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption,self.max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
                                    
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, index):    
        
        image_path = os.path.join(self.image_root, self.ann[index]['image'])        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  

        return image, index
      
        

class pretrain_dataset(Dataset):
    def __init__(self, ann_file, transform, max_words=30):        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.max_words = max_words
        
        
    def __len__(self):
        return len(self.ann)
    

    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        if type(ann['caption']) == list:
            caption = pre_caption(random.choice(ann['caption']), self.max_words)
        else:
            caption = pre_caption(ann['caption'], self.max_words)
      
        image = Image.open(ann['image']).convert('RGB')   
        image = self.transform(image)
                
        return image, caption
            

class pretrain_dataset3d(Dataset):
    def __init__(self, ann_file, transform_pet, transform_mri, dataset='train'):
        """
        Modified to correctly load data from the paths specified in your config file.
        """
        self.ann = []
        # Ensure we handle both list and string inputs from config
        if isinstance(ann_file, list):
            for f in ann_file:
                print(f"Loading 3D Pretrain Data from: {f}") # Debug print
                self.ann += json.load(open(f, 'r'))
        else:
            print(f"Loading 3D Pretrain Data from: {ann_file}") # Debug print
            self.ann = json.load(open(ann_file, 'r'))

        self.transform_pet = transform_pet
        self.transform_mri = transform_mri

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]

        # 1. Load MRI
        mri = nib.load(ann['mri']).get_fdata().astype(np.float32)
        
        # Robust Normalization
        if mri.max() > mri.min():
            mri = (mri - mri.min()) / (mri.max() - mri.min())
        
        # Resize to 128x128x128 (Matching Pretrain3D.yaml)
        x, y, z = mri.shape
        target_res = (128, 128, 128)
        mri = zoom(mri, (target_res[0]/x, target_res[1]/y, target_res[2]/z), order=1)
        
        # Add channel dimension: [D, H, W] -> [1, D, H, W]
        mri = mri[None, ...]

        # 2. Load PET (Tau)
        tau_pet = nib.load(ann['tau_pet']).get_fdata().astype(np.float32)
        
        # Robust Normalization
        if tau_pet.max() > tau_pet.min():
            tau_pet = (tau_pet - tau_pet.min()) / (tau_pet.max() - tau_pet.min())

        x, y, z = tau_pet.shape
        tau_pet = zoom(tau_pet, (target_res[0]/x, target_res[1]/y, target_res[2]/z), order=1)
        tau_pet = tau_pet[None, ...]

        # 3. Apply Augmentations
        if self.transform_mri is not None:
            mri = self.transform_mri(mri)
        
        if self.transform_pet is not None:
            tau_pet = self.transform_pet(tau_pet)

        # Return 0 as dummy label for pretraining
        return mri, tau_pet, 0


class adni_cls_dataset3d(Dataset):
    def __init__(self, ann_file, transform_pet, transform_mri):
        self.ann = []
        if isinstance(ann_file, list):
            for f in ann_file:
                print(f"Loading ADNI Class Data from: {f}") # Debug print
                self.ann += json.load(open(f, 'r'))
        else:
            print(f"Loading ADNI Class Data from: {ann_file}") # Debug print
            self.ann = json.load(open(ann_file, 'r'))

        self.transform_pet = transform_pet
        self.transform_mri = transform_mri
        
    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):    
        
        ann = self.ann[index]

        # 1. Load MRI
        mri = nib.load(ann['mri']).get_fdata().astype(np.float32)
        
        x, y, z = mri.shape
        # Resize to 128x96x128 (Matches train_ADNI.yaml)
        mri = zoom(mri, (128/x, 96/y, 128/z), order=1)[None, ...]
        
        # Standardize
        if mri.std() > 0:
            mri = (mri - mri.mean()) / mri.std()

        # 2. Load PET
        tau_pet = nib.load(ann['tau_pet']).get_fdata().astype(np.float32)
        
        x, y, z = tau_pet.shape
        tau_pet = zoom(tau_pet, (128/x, 96/y, 128/z), order=1)[None, ...]

        # 3. Handle Reference Normalization with Fallback
        ref_file = '/'.join(ann['tau_pet'].split('/')[:-1]) + '/km_inferior.ref.tac.dat'
        
        if os.path.exists(ref_file):
            try:
                inferior_cerebellum = np.loadtxt(ref_file).astype(np.float32)
                tau_pet = tau_pet / inferior_cerebellum
            except:
                if tau_pet.max() > tau_pet.min():
                    tau_pet = (tau_pet - tau_pet.min()) / (tau_pet.max() - tau_pet.min())
        else:
            if tau_pet.max() > tau_pet.min():
                tau_pet = (tau_pet - tau_pet.min()) / (tau_pet.max() - tau_pet.min())

        label = float(ann['cdr'])
        # Map CDR: 0->0, 0.5->1, >=1.0->2
        label = 0 if label == 0 else (1 if label == 0.5 else 2)

        if self.transform_mri is not None:
            mri = self.transform_mri(mri)
        if self.transform_pet is not None:
            tau_pet = self.transform_pet(tau_pet)

        return mri, tau_pet, label


import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nb


class MMDataset(Dataset):
    def __init__(self,df,data_name_list):
        # df: df load from csv file
        # data_name_list: list of col names in csv for dadta file loading
        self.df = df
        self.data_name_list = data_name_list
    def __len__(self):
        return self.df['sub'].shape[0]
    
    def __getitem__(self,idx):
        row = self.df.iloc[idx]
        data_list = []
        for i in self.data_name_list:
            if row[i].split('.')[-1] == 'raw':
                data_list.append(np.fromfile(row[i],dtype=np.float32).reshape(1,-1))
            else:
                data_list.append(nb.freesurfer.io.read_morph_data(row[i]).reshape(1,-1))
        x = np.vstack(data_list)
        # split the first 2 and last 2 columns
        mri = x[:2,:]
        pet = x[-2:,:]
        y = int(row['diag_label'])
        return mri, pet, y