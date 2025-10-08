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
from utils.utils import sdf_to_mask 
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
        self.deform_base_dir = 'dataset/cvpr_final/base_mask_extra'
        self.deform_base_masks = [os.path.join(self.deform_base_dir, f) for f in os.listdir(self.deform_base_dir) if f.endswith('.png') and 'mask' in f]
        self.deform_base_sdfs = [os.path.join(self.deform_base_dir, 'sdf',f) for f in os.listdir(os.path.join(self.deform_base_dir,'sdf')) if f.endswith('.npy')]
        self.deform_base_masks.sort()
        self.deform_base_sdfs.sort()


    def _get_label(self, mask):
        return mask.split('/')[-1].split('_')[0]
    
    def _create_label_to_index_map(self, labels):
        unique_labels = list(set(labels))
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        return label_to_index
        
    def __len__(self):
        return len(self.deform_base_sdfs)
    
    def __getitem__(self, idx):
        mask = cv2.imread(self.deform_base_masks[idx], cv2.IMREAD_GRAYSCALE)
        base_name = self.deform_base_masks[idx].split('/')[-1].split('.')[0]
        mask = self.transform(mask)
        sdf_path = os.path.join(self.deform_base_dir, 'sdf', base_name+'_sdf.npy')
        sdf = np.load(sdf_path)
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
        self.all_deform.sort()
        # add a extra data to all_deform in the first position
        self.all_deform.insert(0, "dataset/deformation_cvpr/deform_shape/deformed_raw_0.obj")
    
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        deform_file = self.all_deform[idx]
        deform_mesh = load_obj(deform_file)
        basename = deform_file.split('/')[-1].split('.')[0]
        canonical_idx = basename.split('_')[0]
        canonical_file = os.path.join(self.root_dir, f'{canonical_idx}_canonical.obj')
        if idx == 0:
            canonical_file = "dataset/deformation_cvpr/base_shape/base_0.obj"
        canonical_mesh = load_obj(canonical_file)
        canonical_points = canonical_mesh[0]
        canonical_faces = canonical_mesh[1][0]
        occ_canonical = gutils.points_to_occ(self.grid_points, canonical_points, res=self.resolution)
        deformed_points = deform_mesh[0]
        # fix random choice
        np.random.seed(0)
        if deformed_points.shape[0]> 3000:
            choice = np.random.choice(deformed_points.shape[0], 3000, replace=False)
            deformed_points = deformed_points[choice]
        data_dict = {
            'canonical_points': canonical_points,
            'deform_points': deformed_points,
            'canonical_faces': canonical_faces,
            'occ_canonical': occ_canonical,
            'idx': idx
        }
        return data_dict
    def get_loader(self, batch_size, shuffle=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)
        
class DeformDataset(Dataset):
    def __init__(self, root_dir = 'dataset/deformation_cvpr_new'):
        self.wo_regis = False
        if self.wo_regis:
            train_dir  = os.path.join(root_dir, 'deform_train')
        else:
            train_dir = os.path.join(root_dir, 'deform_nonrigid_regis')
        base_mesh_dir = os.path.join(root_dir, 'base_shape')
        img_dir = os.path.join(root_dir, 'base_img')
        self.all_deform = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.ply') and 'rigis' in f]
        self.all_base = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.ply') and 'base' in f]
        self.all_basemesh = [os.path.join(base_mesh_dir, f) for f in os.listdir(base_mesh_dir) if f.endswith('.obj')]
        self.all_rgb = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.png')]
        self.all_basemesh.sort()
        self.all_base.sort()
        self.all_deform.sort()
        self.all_rgb.sort()
        
        
    def __len__(self):
        return len(self.all_deform)
    
    def __getitem__(self, idx):
        base_file = self.all_base[idx]
        deform_file = self.all_deform[idx]
        base_mesh = load_ply(base_file)
        deform_mesh = load_ply(deform_file)
        mesh = load_obj(self.all_basemesh[idx])
        # canonical_points = base_mesh[0]
        deformed_points = deform_mesh[0]
        canonical_face = mesh[1][0]
        canonical_points = mesh[0]  
        
        # read img 
        img_name = self.all_rgb[idx]
        img = Image.open(self.all_rgb[idx]).convert('RGB')
        img = img.resize((256, 256))
        img = torch.Tensor(np.array(img, dtype=np.uint8).transpose((2, 0, 1)))

        # TODO: add bone
        # bone = np.random.rand(1000, 10)
        data_dict = {
            'canonical_points': canonical_points,
            'deform_points': deformed_points,
            'canonical_faces': canonical_face,
            'rgb':img,
            'rgb_name': img_name,
            'idx': idx
        }
        return data_dict
    
    def get_loader(self, batch_size, shuffle=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

class DisplacementDataset(Dataset):
    def __init__(self,
                 n_supervision_points_face=2000,
                 ):
        self.n_supervision_points_face = n_supervision_points_face
        self.all_sample = []
        self.root_dir = 'dataset/deformation_cvpr_new/deform_nonrigid_regis'
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            for filename in filenames:
                if filename.endswith('.npy'):
                    self.all_sample.append(os.path.join(dirpath, filename))
        self.all_sample.sort()
    def __len__(self):
        return len(self.all_sample)     
    
    def __getitem__(self, index):
        filename = self.all_sample[index]
        name = os.path.basename(filename)
        # shape_index = name.split('_')[0]
        deform_index = name.split('_')[1]
        trainfile = np.load(self.all_sample[index], allow_pickle=True)
        valid = np.logical_not(np.any(np.isnan(trainfile), axis=-1))
        point_corresp = trainfile[valid,:].astype(np.float32)
        # subsample points for supervision
        sup_idx = np.random.randint(0, point_corresp.shape[0], self.n_supervision_points_face)
        sup_point_neutral = point_corresp[sup_idx,:3]
        sup_posed = point_corresp[sup_idx,3:] 
        # visualize neutral and  3d posed points
        neutral = sup_point_neutral
        pose = sup_posed
        # chamfer distance between neutral and posed
        return {
            'points_neutral': neutral,
            'points_posed': pose,
            'deform_idx': index  
        }
    def get_loader(self, batch_size, shuffle=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

class SemanticSegmentationDataset(Dataset):
    def __init__(self, 
                 img_paths, 
                 mask_paths=None,
                 vein_paths=None,
                 size=(256, 256),
                 mode='binary',
                 normalize=None):
        """
        Example semantic segmentation Dataset class.
        Run once when instantiating the Dataset object.
        If you want to use it for binary semantic segmentation, 
        please select the mode as 'binary'. For multi-class, enter 'multi'.
        example_data/
            └── /images/
                    └── 0001.png
                    └── 0002.png
                    └── 0003.png
                    └── ...
                /masks/
                    └── 0001_mask.png
                    └── 0002_mask.png
                    └── 0003_mask.png
                    └── ...
        img_paths : str
            The file path indicating the main directory that contains only images.
        mask_paths : str, default=None
            The file path indicating the main directory that contains only 
            ground truth images.
        size : tuple, default=(256, 256)
            Enter the (width, height) values into a tuple for resizing the data.
        mode : str, default='binary'
            Choose how the DataSet object should generate data. 
            Enter 'binary' for binary masks.
        normalize : orchvision.transforms.Normalize, default=None
            Normalize a tensor image with mean and standard deviation. 
            This transform does not support PIL Image.
        """
        self.img_paths = [os.path.join(img_paths,f) for f in os.listdir(img_paths) if f.endswith('.jpeg') and 'bot' in f]
        self.mask_paths = [os.path.join(mask_paths,f) for f in os.listdir(mask_paths) if f.endswith('.png') and 'bot' in f] if mask_paths is not None else mask_paths
        self.vein_paths = [os.path.join(vein_paths,f) for f in os.listdir(vein_paths) if f.endswith('.png') and 'bot' in f] if vein_paths is not None else vein_paths
        self.img_paths.sort()
        self.mask_paths.sort()
        self.vein_paths.sort()
        # self.mask_paths = self._get_file_dir(mask_paths) if mask_paths is not None else mask_paths
        # self.vein_paths = self._get_file_dir(vein_paths) if vein_paths is not None else vein_paths
        self.size = size
        self.mode = mode
        self.normalize = None
        self.augmentation = True
        
    def __len__(self):
        """
        Returns the number of samples in our dataset.
        Returns
        -------    
        num_datas : int    
            Number of datas.
        """
        return len(self.img_paths)
    
    def __getitem__(self, index):
        """
        Loads and returns a sample from the dataset at 
        the given index idx. Based on the index, it 
        identifies the image’s location on disk, 
        converts that to a tensor using read_image, 
        retrieves the corresponding label from the 
        ground truth data in self.mask_paths, calls the transform 
        functions on them (if applicable), and returns 
        the tensor image and corresponding label in a tuple.
        Returns
        -------   
        img, mask : torch.Tensor
            The transformed image and its corresponding 
            mask image. If the mask path is None, it 
            will only return the transformed image.
            output_shape_mask: (batch_size, 1, img_size, img_size)
            output_shape_img: (batch_size, 3, img_size, img_size)
        """
        img_path = self.img_paths[index]
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.size[0], self.size[1])) 
        img = torch.Tensor(np.array(img, dtype=np.uint8).transpose((2, 0, 1)))
        if self.mask_paths is not None:
            mask_path = self.mask_paths[index]
            mask = Image.open(mask_path)
            # mask to one channel
            mask = mask.resize((self.size[0], self.size[1])) 
            mask = np.array(mask)

            mask = torch.as_tensor(mask.transpose((2,0,1)), dtype=torch.uint8)
        if self.vein_paths is not None:
            vein_path = self.vein_paths[index]
            vein = Image.open(vein_path)
            vein = vein.resize((self.size[0], self.size[1]))
            vein = np.array(vein)
            vein = torch.as_tensor(vein, dtype=torch.uint8)
    
            if self.augmentation:
                if np.random.rand() > 0.5:
                    img = transforms.functional.hflip(img)
                    mask = transforms.functional.hflip(mask)
                    vein = transforms.functional.hflip(vein)
                if np.random.rand() > 0.5:
                    img = transforms.functional.vflip(img)
                    mask = transforms.functional.vflip(mask)
                    vein = transforms.functional.vflip(vein)
                img = torchvision.transforms.ColorJitter(
                brightness=0.5,
                contrast=0.5,
                hue=0.1, 
                saturation=0.1)(img)
                    
            if self.normalize: 
                img = self.normalize(img.float())
                # mask = self.normalize(mask.float())
            return img, mask[0]/255, vein/255
        else:
           
            return img, mask[0], vein

    def _multi_class_mask(self, mask):
        obj_ids = np.unique(mask)[1:]
        masks = mask == obj_ids[:, None, None]
        return masks

    def _binary_mask(self, mask):
        # input mask shape: (height, width,3)
        
        # mask[:, :][mask[:, :] >= 1] = 1
        # mask[:, :][mask[:, :] < 1] = 0
        binary_mask = np.any(mask >= 1, axis=-1).astype(int)

        binary_mask = np.expand_dims(binary_mask, axis=0)
        # save binary mask
        
        return binary_mask

    def _get_file_dir(self, directory):
        """
        Returns files in the entered directory.
        Parameters
        ----------
        directory : string
            File path.
        Returns
        -------
        directories: list
            All files in the directory.
        """
        def atoi(text):
            return int(text) if text.isdigit() else text
            
        def natural_keys(text):
            return [atoi(c) for c in re.split('(\d+)',text)]

        for roots,dirs,files in os.walk(directory):               
            if files:
                directories = [roots + os.sep + file for file in  files if 'bot' in file]
                directories.sort(key=natural_keys)

        return directories

class DeformLeafDataset(Dataset):
    def __init__(self):
        self.deform_dir = 'dataset/cvpr_final/deform_train_new'
        self.base_dir = 'dataset/cvpr_final/base_shape'
        self.deformshapes = [os.path.join(self.deform_dir, f) for f in os.listdir(self.deform_dir)]
        self.baseshapes = [os.path.join(self.base_dir, f) for f in os.listdir(self.base_dir) if f.endswith('.obj')]
        self.deform_base_dir = 'dataset/cvpr_final/base_mask'
        self.deform_base_masks = [os.path.join(self.deform_base_dir, f) for f in os.listdir(self.deform_base_dir) if f.endswith('.png')]
        self.deform_base_sdfs = [os.path.join(self.deform_base_dir, 'sdf',f) for f in os.listdir(os.path.join(self.deform_base_dir,'sdf')) if f.endswith('.npy')]

        self.deformshapes.sort()
        self.baseshapes.sort()
        self.deform_base_masks.sort()
        # use_sample  = ['leaf_11', 'leaf_5', 'leaf_6','leaf_19','leaf_23','leaf_42','maple4_d8','1-3','1-6','3-3',
        #              '5-2','20','70', '6-3', ]
        # self.deformshapes = [f for f in self.deformshapes if any(substring in f for substring in use_sample)]
    
    def __len__(self):
        return  len(self.deformshapes) 
    
    def get_extra_baselist(self):
        extra_list = []
        return extra_list
    
    def __getitem__(self, index):
        # assert len(self.deformshapes) == len(self.baseshapes)
        deform_file = self.deformshapes[index]
        if deform_file.endswith('.obj'):
            deform_name = deform_file.split('/')[-1].split('.')[0]
            base_name = deform_file.split('/')[-1].split('.')[0]
            deform_mesh = load_obj(deform_file)
        elif deform_file.endswith('.ply'):
            deform_name = deform_file.split('/')[-1].split('.')[0]
            base_name = deform_file.split('/')[-1].split('.')[0].replace('_deform','')
            deform_mesh = load_ply(deform_file)
        base_file = os.path.join(self.base_dir, base_name+'.obj')
        base_mask = os.path.join(self.deform_base_dir,base_name+'.png')
        # search base_name in self.baseshapes to get shape_idx 
        shape_idx = [i for i, f in enumerate(self.deform_base_masks) if base_mask in f]
        deform_points = deform_mesh[0]
        # base_mesh = load_objs_as_meshes([base_file]) 
        data_dict =  {
            'deform_points': deform_points,
            'base_mesh': base_file,
            'idx': index,
            'base_name': base_name,
            'deform_name': deform_name,
            'shape_idx': shape_idx[0]
        }
        return data_dict
    
    def get_loader(self, batch_size, shuffle=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

class DeformLeafDatasetNew(Dataset):
    def __init__(self):
        self.deform_dir = 'dataset/cvpr_final/deform_train_new'
        self.base_dir = 'dataset/cvpr_final/base_mask'
        self.deformshapes = [os.path.join(self.deform_dir, f) for f in os.listdir(self.deform_dir)]
        self.base_masks = [os.path.join(self.base_dir, f) for f in os.listdir(self.base_dir) if f.endswith('.png')]
        self.base_sdf = [os.path.join(self.base_dir, 'sdf', f) for f in os.listdir(os.path.join(self.base_dir, 'sdf'))]
        self.deformshapes.sort()
        self.base_masks.sort()
        self.base_sdf.sort()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
           # transforms.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5],),
            transforms.Resize([128, 128])])
    
    def __len__(self):
        return  len(self.deformshapes) 
    
    def __getitem__(self, index):
        # read base data
        base_mask = self.base_masks[index]
        mask = cv2.imread(base_mask, cv2.IMREAD_GRAYSCALE)
        mask = self.transform(mask)
        sdf = np.load(self.base_sdf[index])
        sdf = sdf.astype(np.float32)
        pts_sample, sdf_sample = self.sdf_sample(sdf, 5000)
        # read deform data
        deform_file = self.deformshapes[index]
        save_name = deform_file.split('/')[-1].split('.')[0]
        if deform_file.endswith('.obj'):
            deform_mesh = load_obj(deform_file)
        else:
            deform_mesh = load_ply(deform_file)
        deform_points = deform_mesh[0]
        deform_samples = deform_points[np.random.choice(deform_points.shape[0], 3000, replace=False)]
        data_dict = {
            'deform_points': deform_points,
            'mask': mask,
            'sdf': sdf_sample,
            'points': pts_sample,
            'idx': index,
            'save_name': save_name
        }
        return data_dict

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
    
    def get_loader(self, batch_size, shuffle=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)


if __name__ == '__main__':
    img_path = 'dataset/2D_Datasets/leaf_vein/rgb'
    mask_path = 'dataset/2D_Datasets/leaf_vein/vein'
    dataset = SemanticSegmentationDataset(img_path, mask_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for i, data in enumerate(dataloader):
        rgb, mask =data

    def process_lvd2021_data():
    # 设置数据的根目录
        root_dir = "vein_segmentation/data/LVD2021"  # 更改为您数据的实际路径

        # 新文件夹路径，用于存储分类后的图像
        output_dirs = {
            "rgb": "dataset/2D_Datasets/leaf_vein/rgb",
            "mask": "dataset/2D_Datasets/leaf_vein/mask",
            "contour": "dataset/2D_Datasets/leaf_vein/contour",
            "vein": "dataset/2D_Datasets/leaf_vein/vein"
        }

    # 创建新目录
        for dir_path in output_dirs.values():
            pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)

        # 遍历每个顶层种类文件夹
        for category_folder in os.listdir(root_dir):
            category_path = os.path.join(root_dir, category_folder)
            
            if os.path.isdir(category_path):
                # 进入 'all' 子文件夹
                all_path = os.path.join(category_path, 'all')
                
                # 遍历每个样本文件夹
                for sample_folder in os.listdir(all_path):
                    sample_path = os.path.join(all_path, sample_folder)
                    
                    # 遍历样本文件夹中的图像
                    for image_name in os.listdir(sample_path):
                        image_path = os.path.join(sample_path, image_name)
                        # 根据文件名确定目标文件夹
                        if "背景" in image_name:
                            target_folder = output_dirs["rgb"]
                        elif "图层 1" in image_name:
                            target_folder = output_dirs["contour"]
                        elif "图层 2" in image_name:
                            target_folder = output_dirs["mask"]
                        elif "图层 3" in image_name:
                            target_folder = output_dirs["vein"]
                        
                        # 构建新的文件名和路径
                        new_name = f"{sample_folder}.png"
                        target_path = os.path.join(target_folder, new_name)
                        
                        # 复制文件到新位置
                        copyfile(image_path, target_path)
                        print(f"File {image_name} copied to {target_path}")
                # print(i, data['mask'].shape, data['label'])
            
            
