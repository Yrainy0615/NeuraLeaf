from torch.utils.data import Dataset, DataLoader
import cv2
import os
from math import log2
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from PIL import Image
import torchvision
import torch
import numpy as np
from data.data_utils import data_processor
# from utils.utils import sdf_to_mask 

def sdf_to_mask(sdf:torch.Tensor,k:float=1):
    mask = 1/ (1+torch.exp(-k*sdf))
    return mask

class BaseShapeDataset(Dataset):
    def __init__(self, data_dir,n_sample):
        self.masks = []
        self.images = []
        self.sdfs = []
        self.sdf_img = []
        self.n_sample = n_sample
        for dirpath, dirnames, filenames in os.walk(data_dir):
            for filename in [f for f in filenames if f.endswith(".JPG")]:
                if 'mask' in filename:
                    self.masks.append(os.path.join(dirpath, filename))
                    sdf_path = os.path.join(dirpath, 'sdf')
                    sdf_file = os.path.join(sdf_path, filename.replace('_mask_aligned.JPG', '_sdf.npy'))
                    self.sdfs.append(sdf_file)
                    sdf_img_path = os.path.join(dirpath, 'output')
                    sdf_img_file = os.path.join(sdf_img_path, filename.replace('.JPG', '_sdf.png'))
                    self.sdf_img.append(sdf_img_file)
                    
                else:
                    self.images.append(os.path.join(dirpath, filename))
        self.labels = [self._get_label(mask) for mask in self.masks]
        self.label_to_index = self._create_label_to_index_map(self.labels)

    def _get_label(self, mask):
        return mask.split('/')[-1].split('_')[0]
    
    def _create_label_to_index_map(self, labels):
        unique_labels = list(set(labels))
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        return label_to_index
        
    def __len__(self):
        return len(self.masks)
    
    def __getitem__(self, idx):
        mask = cv2.imread(self.masks[idx])
        mask = cv2.resize(mask, (128, 128))        
        mask = mask.astype(np.float32)  / 255
        
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512, 512))
        image = image.astype(np.float32) / 127.5 - 1
        
        sdf = np.load(self.sdfs[idx])
        sdf = sdf.astype(np.float32)    
        pts_sample, sdf_sample = self.sdf_sample(sdf, self.n_sample)
        
        sdf_img = cv2.imread(self.sdf_img[idx])
        sdf_img = sdf_img.astype(np.float32) / 255
        
        label = self.masks[idx].split('/')[-1].split('_')[0]
        state = self.masks[idx].split('/')[-1].split('_')[1]
        prompt = f'A photo of a {state} {label} leaf.'
        label_index = self.label_to_index[label]
        data = {
            'hint': mask,
            'jpg': image,
            'label': label_index,
            'idx': idx,
            'sdf_2d': sdf,
            'sdf': sdf_sample,
            'sdf_img': sdf_img,
            'points': pts_sample,
            'txt': prompt
        }
        return data
    def get_loader(self, batch_size, shuffle):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

    def sdf_sample(self, sdf, n_sample):
        size = sdf.shape[0]
        x_coord = np.linspace(0, 1, size).astype(np.float32)
        y_coord = np.linspace(0, 1, size).astype(np.float32)
        xy = np.stack(np.meshgrid(x_coord, y_coord), -1)
        # sample n points with sdf
        idx_x = np.random.choice(size, n_sample, replace=True)
        idx_y = np.random.choice(size, n_sample, replace=True)
        xy_sample = xy[idx_x, idx_y]
        sdf_sample = sdf[idx_x, idx_y]
        return xy_sample, sdf_sample
    
class LeafRGBDataset(Dataset):
    def __init__(self, root_dir='dataset/2D_Datasets', img_size=219, img_list=None):
        self.transforms =  transforms.Compose([
                transforms.Resize([img_size, img_size]), 
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5]),
            ])
        self.processor = data_processor(root_dir)
        self.all_rgb = self.processor.all_rgb
        self.img_list = img_list
    def __getitem__(self, index):
        if self.img_list is not None:
            name = self.img_list[index]
        else: 
            name = self.all_rgb[index]
        img = Image.open(name).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img

    def __len__(self):
        if self.img_list is not None:
            return len(self.img_list)
        else:
            return len(self.all_rgb)
    def get_loader(self, batch_size, shuffle=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

if __name__ == '__main__':
    dataset = BaseShapeDataset('dataset/LeafData', n_sample=1000)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for i, data in enumerate(dataloader):
        sdf = data['sdf_2d']  # (-1,1)
        mask = data['hint']
        mask_rec = sdf_to_mask(sdf,k=50)  #(0,1)

        # create a fig show mask and mask_rec
        grid = make_grid([mask_rec,mask[:,:,:,0]],nrow=1)
        save_image(grid,'test.png')
        
        # print(i, data['mask'].shape, data['label'])
        