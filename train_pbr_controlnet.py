"""
快速验证：PBR Generation with ControlNet + Latent Code
基于现有的 ControlNet 架构
"""
import os
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'submodule', 'controlnet'))

import torch
import argparse
import yaml
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from scripts.models.pbr_controlnet import PBRControlLDM
from scripts.data.pbr_dataset import PBRDataset, SimplePBRDataset
from scripts.utils.pbr_tensorboard_logger import PBRTensorBoardLogger
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


def main():
    parser = argparse.ArgumentParser(description='Train PBR ControlNet')
    parser.add_argument('--gpu', type=int, default=3, help='GPU index')
    parser.add_argument('--config', type=str, 
                       default='scripts/configs/pbr_controlnet.yaml',
                       help='Config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--data_dir', type=str, 
                       default='/mnt/data/2D_Datasets/CGData/CGset',
                       help='Data directory (default: /mnt/data/2D_Datasets/CGData/CGset)')
    parser.add_argument('--latent_code_path', type=str, default=None,
                       help='Path to latent code .pt file')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    
    args = parser.parse_args()
    
    # load config
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
    else:
        print(f"Config file {args.config} not found, using defaults")
        cfg = {}
    
    # device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    

    dataset = PBRDataset(
        data_dir=args.data_dir,
        latent_code_path=args.latent_code_path,
        image_size=512
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0, 
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=False,  
        prefetch_factor=None if torch.cuda.is_available() else 2,  
    )
    
    # create model
    print("Creating model...")
    model_config_path = 'scripts/configs/pbr_controlnet.yaml'
    init_ckpt = 'submodule/controlnet/models/v1-5-pruned.ckpt' 
    if not os.path.exists(model_config_path):
        print(f"Error: Model config {model_config_path} not found")
        print("Please make sure ControlNet submodule is properly set up")
        return
    
    # load base ControlNet model
    model = create_model(model_config_path).cpu()
    
    # if there is a pretrained weight, load it
    if args.resume and os.path.exists(args.resume):
        print(f"Loading checkpoint from {args.resume}")
        model.load_state_dict(load_state_dict(args.resume, location='cpu'))
    else:
        # try to load the initial weights of ControlNet
        if os.path.exists(init_ckpt):
            print(f"Loading initial weights from {init_ckpt}")
            model.load_state_dict(load_state_dict(init_ckpt, location='cpu'), strict=False)
        else:
            print("Warning: No checkpoint found, training from scratch")
    
    # set training parameters
    model.learning_rate = args.lr
    model.sd_locked = True 
    model.only_mid_control = False
    model = model.to(device)
    
    # set logger
    image_logger = ImageLogger(
        batch_frequency=10, 
        max_images=4,  
        disabled=False,  
        log_images_kwargs={"ddim_steps": 50, "sample": True}  
    )
    
    # PBR TensorBoard Logger 
    pbr_tb_logger = PBRTensorBoardLogger(
        batch_frequency=100,  
        max_images=4,  
        log_pbr_channels=True,
        log_separate_channels=True
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/pbr_controlnet/',
        filename='pbr-pixel-{epoch:02d}-{step}',
        every_n_epochs=5,  
        save_top_k=1, 
        monitor='train/loss_simple',
        save_on_train_epoch_end=False,  
        save_last=True,  
    )
    
    # set TensorBoard logger
    from pytorch_lightning.loggers import CSVLogger
    tb_logger = CSVLogger(
        save_dir='logs/pbr_controlnet',
        name='pbr_controlnet_pixel_loss',  # Different log name for pixel-level loss training
    )

    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=[args.gpu] if torch.cuda.is_available() else None,
        precision=32,
        max_epochs=args.num_epochs,
        logger=tb_logger, 
        callbacks=[image_logger, checkpoint_callback], 
        log_every_n_steps=100, 
        enable_progress_bar=True,
        enable_model_summary=False,  
        num_sanity_val_steps=0,  
    )
    
    # train
    print("Starting training...")
    trainer.fit(model, dataloader)
    
    print("Training completed!")


if __name__ == '__main__':
    main()

