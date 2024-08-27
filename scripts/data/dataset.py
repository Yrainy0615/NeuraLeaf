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
from scripts.data.data_utils import data_processor
# from utils.utils import sdf_to_mask 
from scripts.utils import geom_utils as gutils
from pytorch3d.io import load_obj

def sdf_to_mask(sdf:torch.Tensor,k:float=1):
    mask = 1/ (1+torch.exp(-k*sdf))
    return mask

class BaseShapeDataset(Dataset):
    def __init__(self, data_dir,n_sample):
        processor = data_processor(data_dir)
        self.n_sample = n_sample
        self.all_rgb = processor.all_rgb
        self.all_masks = processor.all_masks
        self.extra_mask = processor.extra_mask
        self.healthy_masks = processor.healthy_masks
        self.healthy_rgb = processor.healthy_rgb
        self.diseased_masks = processor.diseased_masks
        self.diseased_rgb = processor.diseased_rgb
        self.base_mask = processor.base_mask
        self.base_sdf = processor.base_sdf
        self.extra_sdf = processor.extra_sdf
        self.healthy_sdf = processor.healthy_sdf
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5],),
            transforms.Resize([128, 128])
        ])
        # self.labels = [self._get_label(mask) for mask in self.masks]
        # self.label_to_index = self._create_label_to_index_map(self.labels)

    def _get_label(self, mask):
        return mask.split('/')[-1].split('_')[0]
    
    def _create_label_to_index_map(self, labels):
        unique_labels = list(set(labels))
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        return label_to_index
        
    def __len__(self):
        return len(self.base_sdf)
    
    def __getitem__(self, idx):
        mask = cv2.imread(self.base_mask[idx], cv2.IMREAD_GRAYSCALE)
        mask = self.transform(mask)
        
        # image = cv2.imread(self.images[idx])
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = cv2.resize(image, (512, 512))
        # image = image.astype(np.float32) / 127.5 - 1
        
        sdf = np.load(self.base_sdf[idx])
        sdf = sdf.astype(np.float32)    
        pts_sample, sdf_sample = self.sdf_sample(sdf, self.n_sample)
        # label = self.base_mask[idx].split('/')[-1].split('_')[0]
        # state = self.base_mask[idx].split('/')[-1].split('_')[1]
        # prompt = f'A photo of a {state} {label} leaf.'
        # label_index = self.label_to_index[label]
        data = {
            'hint': mask,
            # 'label': label_index,
            'idx': idx,
            'sdf_2d': sdf,
            'sdf': sdf_sample,
            'points': pts_sample,
            # 'txt': prompt
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
    def __init__(self, root_dir='dataset/2D_Datasets', img_size=256, img_list=None):
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
        return img, name

    def __len__(self):
        if self.img_list is not None:
            return len(self.img_list)
        else:
            return len(self.all_rgb)
    def get_loader(self, batch_size, shuffle=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)


class LbsDataset(Dataset):
    def __init__(self, root_dir='dataset/deformation_eccv', resolution=128):
        self.root_dir = root_dir
        self.all_deform = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if not 'canonical' in f and f.endswith('.obj')]
        mini = [-.95, -.95, -.95]
        maxi = [0.95, 0.95, 0.95]
        self.resolution = resolution    
        self.grid_points = gutils.create_grid_points_from_bounds(mini, maxi, resolution)
        self.transform = False
    
    def __len__(self):
        return len(self.all_deform)
    
    def __getitem__(self, idx):
        deform_file = self.all_deform[idx]
        deform_mesh = load_obj(deform_file)
        basename = deform_file.split('/')[-1].split('.')[0]
        canonical_idx = basename.split('_')[0]
        canonical_file = os.path.join(self.root_dir, f'{canonical_idx}_canonical.obj')
        canonical_mesh = load_obj(canonical_file)
        canonical_points = canonical_mesh.verts_packed()
        occ_canonical = gutils.points_to_occ(self.grid_points, canonical_points, res=self.resolution)
        data_dict = {
            'canonical_mesh': canonical_mesh,
            'deform_mesh': deform_mesh,
            'occ_canonical': occ_canonical,
            'idx': idx
        }
        return data_dict
        

        


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
        