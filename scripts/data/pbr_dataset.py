"""
PBR Dataset for ControlNet training
支持：mask + latent_code → PBR maps (albedo, normal, displacement)

数据格式：/mnt/data/2D_Datasets/CGData/CGset/
文件命名：{base_name}_{map_type}_{variant_number}.jpg
map_type: albedo, normal, displacement, mask
"""
import os
import cv2
import numpy as np
import torch
import re
from torch.utils.data import Dataset
from torchvision import transforms
from collections import defaultdict


class PBRDataset(Dataset):
    """
    Dataset for PBR generation
    适配 CGData 数据格式：
    - 文件命名：{base_name}_{map_type}_{variant_number}.jpg
    - map_type: albedo, normal, displacement, mask
    """
    def __init__(self, 
                 data_dir='/mnt/data/2D_Datasets/CGData/CGset',
                 latent_code_path=None,
                 image_size=512,
                 normalize_pbr=True,
                 map_types=None):
        """
        Args:
            data_dir: 数据目录（包含所有 map 文件的目录）
            latent_code_path: latent code 的 .pt 文件路径（可选）
            image_size: 图像尺寸
            normalize_pbr: 是否归一化 PBR maps
            map_types: 要加载的 map 类型列表，默认 ['albedo', 'normal', 'displacement', 'mask']
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.normalize_pbr = normalize_pbr
        self.map_types = map_types or ['albedo', 'normal', 'displacement', 'mask']
        
        # 扫描所有文件，构建索引
        print(f"Scanning PBR data from {data_dir}...")
        self.samples = self._scan_data_directory()
        print(f"Found {len(self.samples)} valid samples")
        
        # 加载 latent codes（如果有）
        self.latent_codes = None
        if latent_code_path and os.path.exists(latent_code_path):
            self.latent_codes = torch.load(latent_code_path)
            if isinstance(self.latent_codes, dict):
                # 如果是 dict，尝试找到 'latent' 或 'weight' key
                self.latent_codes = self.latent_codes.get('latent', 
                                                          self.latent_codes.get('weight', None))
            if self.latent_codes is not None:
                print(f"Loaded {len(self.latent_codes)} latent codes from {latent_code_path}")
        
    def _scan_data_directory(self):
        """
        扫描数据目录，找到所有有效的 (base_name, variant) 组合
        返回: list of (base_name, variant_number, file_paths_dict)
        """
        if not os.path.exists(self.data_dir):
            raise ValueError(f"Data directory {self.data_dir} does not exist")
        
        # 文件命名模式：{base_name}_{map_type}_{variant_number}.jpg
        pattern = re.compile(r'^(.+?)_(albedo|normal|displacement|mask)_(\d+)\.(jpg|jpeg|png)$', re.IGNORECASE)
        
        # 按 base_name 和 variant 组织文件
        samples_dict = defaultdict(lambda: defaultdict(dict))
        
        for filename in os.listdir(self.data_dir):
            match = pattern.match(filename)
            if match:
                base_name = match.group(1)
                map_type = match.group(2).lower()
                variant = int(match.group(3))
                ext = match.group(4)
                
                if map_type in self.map_types:
                    file_path = os.path.join(self.data_dir, filename)
                    samples_dict[base_name][variant][map_type] = file_path
        
        # 构建样本列表：只保留有完整 map 的 variant
        samples = []
        for base_name, variants in samples_dict.items():
            for variant, maps in variants.items():
                # 检查是否有所有必需的 map
                has_all_maps = all(map_type in maps for map_type in self.map_types)
                if has_all_maps:
                    samples.append({
                        'base_name': base_name,
                        'variant': variant,
                        'maps': maps
                    })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        maps = sample['maps']
        
        # Load mask
        mask_path = maps['mask']
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Cannot load mask from {mask_path}")
        
        # 转换为 binary mask
        mask = (mask > 127).astype(np.float32)
        mask = (mask * 255).astype(np.uint8)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)  # [H, W, 3]
        mask = mask.astype(np.float32) / 255.0  # [0, 1]
        
        # Load albedo
        albedo_path = maps['albedo']
        albedo = cv2.imread(albedo_path, cv2.IMREAD_COLOR)
        if albedo is None:
            raise ValueError(f"Cannot load albedo from {albedo_path}")
        albedo = cv2.cvtColor(albedo, cv2.COLOR_BGR2RGB)
        albedo = albedo.astype(np.float32) / 255.0  # [0, 1]
        
        # Load normal (通常存储为 RGB，需要转换到 [-1, 1])
        normal_path = maps['normal']
        normal = cv2.imread(normal_path, cv2.IMREAD_COLOR)
        if normal is None:
            raise ValueError(f"Cannot load normal from {normal_path}")
        normal = cv2.cvtColor(normal, cv2.COLOR_BGR2RGB)
        normal = normal.astype(np.float32) / 255.0
        normal = normal * 2.0 - 1.0  # [0, 1] -> [-1, 1]
        
        # Load displacement
        disp_path = maps['displacement']
        disp = cv2.imread(disp_path, cv2.IMREAD_GRAYSCALE)
        if disp is None:
            raise ValueError(f"Cannot load displacement from {disp_path}")
        disp = disp.astype(np.float32) / 255.0  # [0, 1]
        
        # Resize to target size
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        albedo = cv2.resize(albedo, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        normal = cv2.resize(normal, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        disp = cv2.resize(disp, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        disp = np.expand_dims(disp, axis=-1)  # [H, W, 1] - 在 resize 后添加维度
        
        # 组合 PBR maps: [albedo(3) + normal(3) + disp(1)] = 7 channels
        pbr = np.concatenate([albedo, normal, disp], axis=-1)  # [H, W, 7]
        
        # 转换为 tensor
        # 注意：DDPM 的 get_input 期望 [B, H, W, C] 格式（DataLoader 会自动添加 batch 维度）
        # 所以这里返回 [H, W, C] 格式
        mask_tensor = torch.from_numpy(mask).float()  # [H, W, 3]
        pbr_tensor = torch.from_numpy(pbr).float()  # [H, W, 7]
        
        # 归一化 PBR 到 [-1, 1]（如果需要）
        if self.normalize_pbr:
            # Albedo: [0, 1] -> [-1, 1]
            pbr_tensor[:, :, :3] = pbr_tensor[:, :, :3] * 2.0 - 1.0
            # Normal: 已经是 [-1, 1]
            # Displacement: [0, 1] -> [-1, 1]
            pbr_tensor[:, :, 6:7] = pbr_tensor[:, :, 6:7] * 2.0 - 1.0
        
        result = {
            'hint': mask_tensor,  # ControlNet condition [H, W, 3] -> DataLoader 后变成 [B, H, W, 3]
            'jpg': pbr_tensor[:, :, :3],  # Albedo (RGB) [H, W, 3] -> [B, H, W, 3] (用于兼容)
            'pbr': pbr_tensor,  # 完整的 PBR maps [H, W, 7] -> [B, H, W, 7] (albedo + normal + displacement)
            'txt': '',  # 文本条件（ControlNet 需要，但我们用 latent_code 替代）
        }
        
        # 添加 latent code（如果有）
        if self.latent_codes is not None:
            if idx < len(self.latent_codes):
                result['latent_code'] = self.latent_codes[idx]
            else:
                # 如果没有对应的 latent code，用零向量
                result['latent_code'] = torch.zeros(self.latent_codes.shape[1])
        
        return result


class SimplePBRDataset(Dataset):
    """
    简化版数据集：用于快速测试
    如果没有真实的 PBR 数据，可以用这个生成合成数据
    """
    def __init__(self, num_samples=100, image_size=512, latent_dim=256):
        self.num_samples = num_samples
        self.image_size = image_size
        self.latent_dim = latent_dim
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 生成随机 mask [H, W, 3]
        mask = torch.rand(self.image_size, self.image_size, 1)
        mask = (mask > 0.5).float()
        mask = mask.repeat(1, 1, 3)  # [H, W, 3]
        
        # 生成随机 PBR maps（用于测试）[H, W, C]
        albedo = torch.rand(self.image_size, self.image_size, 3) * 2.0 - 1.0
        normal = torch.rand(self.image_size, self.image_size, 3) * 2.0 - 1.0
        normal = torch.nn.functional.normalize(normal, p=2, dim=-1)  # 归一化最后一个维度
        displacement = torch.rand(self.image_size, self.image_size, 1) * 2.0 - 1.0
        
        pbr = torch.cat([albedo, normal, displacement], dim=-1)  # [H, W, 7]
        
        # 随机 latent code
        latent_code = torch.randn(self.latent_dim)
        
        return {
            'hint': mask,  # [H, W, 3]
            'jpg': albedo,  # [H, W, 3] - 先用 albedo 作为 target
            'pbr': pbr,  # [H, W, 7]
            'txt': '',  # 文本条件（ControlNet 需要，但我们用 latent_code 替代）
            'latent_code': latent_code,
        }

