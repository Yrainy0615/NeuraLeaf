from torch.utils.data import Dataset, DataLoader
import cv2
import os
import re
from math import log2
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from PIL import Image
import torchvision
import torch
import numpy as np
from scripts.data.data_utils import data_processor
from scripts.utils import geom_utils as gutils
from pytorch3d.io import load_obj, load_ply, load_objs_as_meshes
from shutil import copyfile
import pathlib
from scripts.data.mesh_process import MeshProcessor 
import warnings
warnings.filterwarnings("ignore")

def sdf_to_mask(sdf:torch.Tensor,k:float=1):
    mask = 1/ (1+torch.exp(-k*sdf))
    return mask

class BaseShapeDataset(Dataset):
    def __init__(self, data_dir,n_sample):
        processor = data_processor(data_dir)
        self.n_sample = n_sample
        self.all_rgb = processor.all_rgb
        self.all_masks = processor.all_masks  # Leaf_RGB
        self.all_sdf = processor.all_sdf
        self.extra_mask = processor.extra_mask # canonical_mask/
        self.extra_sdf = processor.extra_sdf
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5],),
            transforms.Resize([128, 128])
        ])
        self.deform_base_dir = '/mnt/data/cvpr_final/base_mask'
        # Load and sort by base_name (not filename) to ensure alignment with DeformLeafDataset
        mask_files = [f for f in os.listdir(self.deform_base_dir) if f.endswith('.png')]
        sdf_files = [f for f in os.listdir(os.path.join(self.deform_base_dir, 'sdf')) if f.endswith('.npy')]
        
        # Create pairs: (base_name, filename) and sort by base_name
        mask_pairs = [(f.replace('.png', ''), f) for f in mask_files]
        mask_pairs.sort(key=lambda x: x[0])  # Sort by base_name
        
        sdf_pairs = [(f.replace('_sdf.npy', '').replace('.npy', ''), f) for f in sdf_files]
        sdf_pairs.sort(key=lambda x: x[0])  # Sort by base_name
        
        self.deform_base_masks = [os.path.join(self.deform_base_dir, f) for _, f in mask_pairs]
        self.deform_base_sdfs = [os.path.join(self.deform_base_dir, 'sdf', f) for _, f in sdf_pairs]
        # create all base shapes , keep the deform base masks at the first position
        self.all_base_masks = self.deform_base_masks + self.all_masks + self.extra_mask
        self.all_base_sdfs = self.deform_base_sdfs + self.all_sdf + self.extra_sdf
        print(f"Loaded {len(self.all_base_masks)} base shapes, in which {len(self.deform_base_masks)} are deform base shapes")
        
    def __len__(self):
        return len(self.all_base_masks)
    
    def __getitem__(self, idx):
        mask = cv2.imread(self.all_base_masks[idx], cv2.IMREAD_GRAYSCALE)
        mask = self.transform(mask)
        sdf = np.load(self.all_base_sdfs[idx])
        sdf = sdf.astype(np.float32)    
        pts_sample, sdf_sample = self.sdf_sample(sdf, self.n_sample)
        data = {
            'hint': mask,
            'idx': idx,
            'sdf_2d': sdf,
            'sdf': sdf_sample,
            'points': pts_sample,
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
        

class DeformLeafDataset(Dataset):
    def __init__(self):
        self.deform_dir = '/mnt/data/cvpr_final/deform_train'
        self.base_dir = '/mnt/data/cvpr_final/base_shape'
        self.deform_base_dir = '/mnt/data/cvpr_final/base_mask'
        
        # Load all files
        deform_files = [f for f in os.listdir(self.deform_dir) if f.endswith(('.obj', '.ply'))]
        base_mask_files = [f for f in os.listdir(self.deform_base_dir) if f.endswith('.png')]
        base_shape_files = [f for f in os.listdir(self.base_dir) if f.endswith('.obj')]
        
        # Extract base names from deform files for sorting
        def get_base_name_from_deform(filename):
            """Extract base name from deform filename"""
            if filename.endswith('.obj'):
                return filename.replace('.obj', '')
            elif filename.endswith('.ply'):
                return filename.replace('_deform.ply', '').replace('.ply', '')
            return filename
        
        def get_base_name_from_mask(filename):
            """Extract base name from mask filename"""
            return filename.replace('.png', '')
        
        # Create pairs: (base_name, deform_file) and sort by base_name
        deform_pairs = [(get_base_name_from_deform(f), f) for f in deform_files]
        deform_pairs.sort(key=lambda x: x[0])  # Sort by base_name
        
        # Create pairs: (base_name, mask_file) and sort by base_name
        mask_pairs = [(get_base_name_from_mask(f), f) for f in base_mask_files]
        mask_pairs.sort(key=lambda x: x[0])  # Sort by base_name
        
        # Build sorted lists
        self.deformshapes = [os.path.join(self.deform_dir, f) for _, f in deform_pairs]
        self.deform_base_masks = [os.path.join(self.deform_base_dir, f) for _, f in mask_pairs]
        
        # Build base_name to index mapping for shape_idx lookup
        self.base_name_to_mask_idx = {base_name: idx for idx, (base_name, _) in enumerate(mask_pairs)}
        
        # Sort base shapes by base_name to match deformshapes order
        # Create a mapping from base_name to base_shape file
        base_shape_dict = {}
        for f in base_shape_files:
            base_name = f.replace('.obj', '')
            base_shape_dict[base_name] = f
        
        # Build baseshapes in the same order as deformshapes
        self.baseshapes = []
        for base_name, _ in deform_pairs:
            self.baseshapes.append(os.path.join(self.base_dir, base_shape_dict[base_name]))
        #     else:
        #         print(f"Warning: base shape not found for base_name {base_name}")
        
        # # Add any remaining base shapes that are not in deform pairs (if any)
        # remaining_base_names = set(base_shape_dict.keys()) - {base_name for base_name, _ in deform_pairs}
        # if remaining_base_names:
        #     remaining_files = sorted([base_shape_dict[name] for name in remaining_base_names])
        #     self.baseshapes.extend([os.path.join(self.base_dir, f) for f in remaining_files])
        
        print(f"DeformLeafDataset: {len(self.deformshapes)} deform samples, {len(self.deform_base_masks)} base masks")

    
    def __len__(self):
        return  len(self.deformshapes) 
    
    def get_extra_baselist(self):
        extra_list = []
        return extra_list
    
    def __getitem__(self, index):
        """
        Get deformation sample.
        
        Note: index should align with BaseShapeDataset's first N_deform samples.
        Both datasets are sorted by base_name, so index directly corresponds to shape_idx.
        """
        deform_file = self.deformshapes[index]
        
        # Extract base_name from deform filename
        if deform_file.endswith('.obj'):
            deform_name = os.path.basename(deform_file).replace('.obj', '')
            base_name = deform_name
            deform_mesh = load_obj(deform_file)
        elif deform_file.endswith('.ply'):
            deform_name = os.path.basename(deform_file).replace('.ply', '')
            base_name = deform_name.replace('_deform', '')
            deform_mesh = load_ply(deform_file)
        
        base_file = os.path.join(self.base_dir, base_name + '.obj')
        
        # Get shape_idx: since both datasets are sorted by base_name,
        # we can directly use the mapping or index
        if base_name in self.base_name_to_mask_idx:
            shape_idx = self.base_name_to_mask_idx[base_name]
        else:
            # Fallback: search in deform_base_masks
            base_mask = os.path.join(self.deform_base_dir, base_name + '.png')
            shape_idx = [i for i, f in enumerate(self.deform_base_masks) if base_mask in f]
            if len(shape_idx) > 0:
                shape_idx = shape_idx[0]
            else:
                # If not found, use index (assuming alignment)
                shape_idx = index
                print(f"Warning: base_name {base_name} not found in masks, using index {index}")
        
        deform_points = deform_mesh[0]
        
        data_dict = {
            'deform_points': deform_points,
            'base_mesh': base_file,
            'idx': index,
            'base_name': base_name,
            'deform_name': deform_name,
            'shape_idx': shape_idx  # This should match the index in BaseShapeDataset (0 to N_deform-1)
        }
        return data_dict
    
    def get_loader(self, batch_size, shuffle=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)
            
            
class EncoderDataset(Dataset):
    """Dataset for training encoder with point clouds and GT codes."""
    
    def __init__(self, deform_dataset, shape_codes, deform_codes, resolution=128, device='cpu'):

        self.deform_dataset = deform_dataset
        self.resolution = resolution
        self.shape_codes = shape_codes
        self.deform_codes = deform_codes
        
    def __len__(self):
        return len(self.deform_dataset)
    
    def __getitem__(self, idx):
        # Get base-deform pair from DeformLeafDataset
        data = self.deform_dataset[idx]
        deform_points = data['deform_points']  # (N, 3) - deformed point cloud
        shape_idx = data['shape_idx']  # index for base shape
        deform_idx = data['idx']  # index for deform sample
        
        # Convert point cloud to SDF grid (voxel)
        sdf_grid = gutils.points_to_voxel(deform_points, resolution=self.resolution)
        
        # Get GT codes using indices from DeformLeafDataset
        gt_shape_code = self.shape_codes[shape_idx]  # (code_dim,)
        gt_deform_code = self.deform_codes[deform_idx]  # (code_dim,)
        
        return {
            'sdf_grid': sdf_grid,
            'gt_shape_code': gt_shape_code,
            'gt_deform_code': gt_deform_code,
            'shape_idx': shape_idx,
            'deform_idx': deform_idx,
            'deform_points': deform_points
        }
    
    def get_loader(self, batch_size, shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=4)