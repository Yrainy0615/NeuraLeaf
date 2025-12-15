"""
PBR TensorBoard Logger
记录生成的 PBR maps 和 GT maps 到 TensorBoard
"""
import os
import torch
import torchvision
from pytorch_lightning.callbacks import Callback
try:
    from pytorch_lightning.utilities.rank_zero import rank_zero_only
except ImportError:
    from pytorch_lightning.utilities.distributed import rank_zero_only


class PBRTensorBoardLogger(Callback):
    """
    专门用于记录 PBR maps 的 TensorBoard Logger
    记录：生成的 maps、GT maps、mask 等
    """
    def __init__(self, 
                 batch_frequency=100, 
                 max_images=4,
                 log_pbr_channels=True,
                 log_separate_channels=True):
        """
        Args:
            batch_frequency: 每 N 个 batch 记录一次
            max_images: 最多记录多少张图片
            log_pbr_channels: 是否记录 PBR 通道（albedo, normal, displacement）
            log_separate_channels: 是否分别记录每个通道
        """
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.log_pbr_channels = log_pbr_channels
        self.log_separate_channels = log_separate_channels
        
    @rank_zero_only
    def log_to_tensorboard(self, pl_module, batch, batch_idx, split="train"):
        """记录到 TensorBoard"""
        if not hasattr(pl_module, 'logger') or pl_module.logger is None:
            return
        
        # 检查是否应该记录
        if batch_idx % self.batch_freq != 0:
            return
        
        # 获取 batch 数据
        hint = batch.get('hint', None)  # mask
        pbr_gt = batch.get('pbr', None)  # GT PBR maps [B, 7, H, W]
        jpg_gt = batch.get('jpg', None)  # GT RGB (albedo) [B, 3, H, W]
        
        # 限制图片数量
        N = min(self.max_images, hint.shape[0] if hint is not None else 0)
        if N == 0:
            return
        
        # 记录 mask
        if hint is not None:
            mask_to_log = hint[:N].clone()
            # hint 格式是 [B, H, W, 3]，需要转换为 [B, 3, H, W]
            if len(mask_to_log.shape) == 4 and mask_to_log.shape[-1] == 3:
                mask_to_log = mask_to_log.permute(0, 3, 1, 2)  # [B, H, W, 3] -> [B, 3, H, W]
            elif len(mask_to_log.shape) == 4 and mask_to_log.shape[1] == 3:
                # 已经是 [B, 3, H, W] 格式
                pass
            else:
                # 如果是单通道，扩展到 3 通道
                if len(mask_to_log.shape) == 3:
                    mask_to_log = mask_to_log.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
                if mask_to_log.shape[1] == 1:
                    mask_to_log = mask_to_log.repeat(1, 3, 1, 1)  # [B, 1, H, W] -> [B, 3, H, W]
            
            # 确保是 float 类型且在 [0, 1] 范围
            mask_to_log = mask_to_log.float()
            if mask_to_log.max() > 1.0:
                mask_to_log = mask_to_log / 255.0
            mask_to_log = torch.clamp(mask_to_log, 0.0, 1.0)
            
            grid_mask = torchvision.utils.make_grid(mask_to_log, nrow=min(4, N), normalize=False)
            pl_module.logger.experiment.add_image(
                f'{split}/mask_condition',
                grid_mask,
                pl_module.global_step
            )
        
        # 记录 GT PBR maps
        if pbr_gt is not None:
            pbr_gt = pbr_gt[:N].clone()
            # pbr_gt 格式是 [B, H, W, 7]，需要转换为 [B, 7, H, W]
            if len(pbr_gt.shape) == 4 and pbr_gt.shape[-1] == 7:
                pbr_gt = pbr_gt.permute(0, 3, 1, 2)  # [B, H, W, 7] -> [B, 7, H, W]
            
            # Albedo (前 3 通道)
            if pbr_gt.shape[1] >= 3:
                albedo_gt = pbr_gt[:, :3, :, :].float()
                # 从 [-1, 1] 转换到 [0, 1]
                albedo_gt = (albedo_gt + 1.0) / 2.0
                albedo_gt = torch.clamp(albedo_gt, 0.0, 1.0)
                grid_albedo = torchvision.utils.make_grid(albedo_gt, nrow=min(4, N), normalize=False)
                pl_module.logger.experiment.add_image(
                    f'{split}/gt_albedo',
                    grid_albedo,
                    pl_module.global_step
                )
            
            # Normal (第 4-6 通道)
            if pbr_gt.shape[1] >= 6:
                normal_gt = pbr_gt[:, 3:6, :, :].float()
                # Normal 已经是 [-1, 1]，转换到 [0, 1] 用于可视化
                normal_gt = (normal_gt + 1.0) / 2.0
                normal_gt = torch.clamp(normal_gt, 0.0, 1.0)
                grid_normal = torchvision.utils.make_grid(normal_gt, nrow=min(4, N), normalize=False)
                pl_module.logger.experiment.add_image(
                    f'{split}/gt_normal',
                    grid_normal,
                    pl_module.global_step
                )
            
            # Displacement (第 7 通道)
            if pbr_gt.shape[1] >= 7:
                disp_gt = pbr_gt[:, 6:7, :, :].float()
                # Displacement 从 [-1, 1] 转换到 [0, 1]
                disp_gt = (disp_gt + 1.0) / 2.0
                disp_gt = torch.clamp(disp_gt, 0.0, 1.0)
                # 扩展到 3 通道用于可视化
                disp_gt = disp_gt.repeat(1, 3, 1, 1)
                grid_disp = torchvision.utils.make_grid(disp_gt, nrow=min(4, N), normalize=False)
                pl_module.logger.experiment.add_image(
                    f'{split}/gt_displacement',
                    grid_disp,
                    pl_module.global_step
                )
        
        # 尝试获取模型生成的图像
        if hasattr(pl_module, 'log_images') and callable(pl_module.log_images):
            is_train = pl_module.training
            if is_train:
                pl_module.eval()
            
            try:
                with torch.no_grad():
                    # 调用模型的 log_images 方法
                    log_dict = pl_module.log_images(batch, N=N, split=split)
                    
                    # 记录生成的图像
                    for key, images in log_dict.items():
                        if isinstance(images, torch.Tensor):
                            images = images[:N].detach().cpu()
                            # 确保在 [0, 1] 范围内
                            if images.min() < 0:
                                images = (images + 1.0) / 2.0
                            images = torch.clamp(images, 0.0, 1.0)
                            
                            # 处理不同通道数
                            if images.shape[1] == 1:
                                images = images.repeat(1, 3, 1, 1)
                            elif images.shape[1] > 3:
                                images = images[:, :3, :, :]  # 只取前 3 通道
                            
                            grid = torchvision.utils.make_grid(images, nrow=min(4, N))
                            pl_module.logger.experiment.add_image(
                                f'{split}/generated_{key}',
                                grid,
                                pl_module.global_step
                            )
            except Exception as e:
                print(f"Warning: Failed to log generated images: {e}")
            
            if is_train:
                pl_module.train()
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """训练 batch 结束时记录"""
        self.log_to_tensorboard(pl_module, batch, batch_idx, split="train")
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """验证 batch 结束时记录"""
        self.log_to_tensorboard(pl_module, batch, batch_idx, split="val")

