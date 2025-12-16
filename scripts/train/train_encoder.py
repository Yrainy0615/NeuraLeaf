import os
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import torch
import torch.optim as optim
import argparse
import yaml
from torch.utils.data import DataLoader
from scripts.models.encoder import ShapeEncoder
from scripts.data.dataset import DeformLeafDataset, EncoderDataset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


class EncoderTrainer:
    def __init__(self, cfg, device, args, encoder_dataset, resolution=128):
        self.cfg = cfg['Trainer']
        self.device = device
        self.args = args            
        self.dataset = encoder_dataset
        self.shape_codes = encoder_dataset.shape_codes
        self.deform_codes = encoder_dataset.deform_codes
        self.dataloader = encoder_dataset.get_loader(
            batch_size=self.cfg['batch_size'], 
            shuffle=True
        )
        
        # Create encoders
        shape_code_dim = self.shape_codes.shape[1]
        deform_code_dim = self.deform_codes.shape[1]        
        self.shape_encoder = ShapeEncoder(
            code_dim=shape_code_dim,
            res=resolution,
            output_dim=shape_code_dim
        ).to(device)
        
        self.deform_encoder = ShapeEncoder(
            code_dim=deform_code_dim,
            res=resolution,
            output_dim=deform_code_dim
        ).to(device)
        
        # Optimizers
        lr_encoder = self.cfg.get('LR_encoder', 0.001)
        self.optimizer_shape = optim.Adam(
            self.shape_encoder.parameters(), 
            lr=lr_encoder
        )
        self.optimizer_deform = optim.Adam(
            self.deform_encoder.parameters(), 
            lr=lr_encoder
        )
        
        # TensorBoard
        if args.use_tensorboard:
            log_dir = os.path.join('logs', 'tensorboard', f"encoder_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
            self.global_step = 0
        else:
            self.writer = None
    
    def train_step(self, batch):
        """Training step."""
        sdf_grid = batch['sdf_grid'].to(self.device)  # (B, res, res, res)
        gt_shape_code = batch['gt_shape_code'].to(self.device)  # (B, code_dim)
        gt_deform_code = batch['gt_deform_code'].to(self.device)  # (B, code_dim)
        
        # Forward pass
        # Encoder output: (B, 1, code_dim) -> squeeze to (B, code_dim)
        pred_shape_code = self.shape_encoder(sdf_grid).squeeze(1)  # (B, code_dim)
        pred_deform_code = self.deform_encoder(sdf_grid).squeeze(1)  # (B, code_dim)
        
        # Loss: L2 distance between predicted and GT codes
        loss_shape = torch.nn.functional.mse_loss(pred_shape_code, gt_shape_code)
        loss_deform = torch.nn.functional.mse_loss(pred_deform_code, gt_deform_code)
        
        # Total loss
        loss_total = loss_shape + loss_deform
        
        # Backward pass
        self.optimizer_shape.zero_grad()
        self.optimizer_deform.zero_grad()
        loss_total.backward()
        self.optimizer_shape.step()
        self.optimizer_deform.step()
        
        loss_dict = {
            'loss_shape': loss_shape.item(),
            'loss_deform': loss_deform.item(),
            'loss_total': loss_total.item()
        }
        
        # Log to TensorBoard
        if self.writer is not None:
            for key, value in loss_dict.items():
                self.writer.add_scalar(f'Train/{key}', value, self.global_step)
            self.global_step += 1
        
        return loss_dict
    
    def save_checkpoint(self, checkpoint_path, save_name, epoch):
        """Save checkpoint."""
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        save_path = os.path.join(checkpoint_path, save_name)
        torch.save({
            'epoch': epoch,
            'shape_encoder': self.shape_encoder.state_dict(),
            'deform_encoder': self.deform_encoder.state_dict(),
            'optimizer_shape': self.optimizer_shape.state_dict(),
            'optimizer_deform': self.optimizer_deform.state_dict(),
        }, save_path)
        print(f'Checkpoint saved to {save_path}')
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint."""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.shape_encoder.load_state_dict(checkpoint['shape_encoder'])
            self.deform_encoder.load_state_dict(checkpoint['deform_encoder'])
            self.optimizer_shape.load_state_dict(checkpoint['optimizer_shape'])
            self.optimizer_deform.load_state_dict(checkpoint['optimizer_deform'])
            print(f'Checkpoint loaded from {checkpoint_path}')
            return checkpoint.get('epoch', 0)
        return 0
    
    def reduce_lr(self, epoch, factor=0.9):
        """Reduce learning rate."""
        if epoch % self.cfg.get('lr_decay_interval', 100) == 0:
            for param_group in self.optimizer_shape.param_groups:
                param_group['lr'] *= factor
            for param_group in self.optimizer_deform.param_groups:
                param_group['lr'] *= factor
    
    def train(self):
        """Main training loop."""
        num_epochs = self.cfg['num_epochs']
        start_epoch = 0
        
        if self.args.continue_train:
            checkpoint_path = os.path.join(self.cfg['checkpoint_path'], 'latest_encoder.pth')
            start_epoch = self.load_checkpoint(checkpoint_path)
        
        for epoch in range(start_epoch, num_epochs):
            sum_loss_dict = {'loss_shape': 0.0, 'loss_deform': 0.0, 'loss_total': 0.0}
            
            for batch in self.dataloader:
                loss_dict = self.train_step(batch)
                for key in sum_loss_dict:
                    sum_loss_dict[key] += loss_dict[key]
            
            # Average losses
            n_batches = len(self.dataloader)
            for key in sum_loss_dict:
                sum_loss_dict[key] /= n_batches
            
            # Reduce learning rate
            self.reduce_lr(epoch)
            
            # Print progress
            print(f'Epoch {epoch}/{num_epochs}: ', end='')
            for key, value in sum_loss_dict.items():
                print(f'{key}: {value:.6f} ', end='')
            print()
            
            # Save checkpoint
            if epoch % self.cfg.get('ckpt_interval', 50) == 0:
                self.save_checkpoint(
                    self.cfg['checkpoint_path'], 
                    f'epoch_{epoch}_encoder.pth', 
                    epoch
                )
            
            # Save latest checkpoint
            self.save_checkpoint(
                self.cfg['checkpoint_path'], 
                'latest_encoder.pth', 
                epoch
            )
        
        if self.writer is not None:
            self.writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Encoder')
    parser.add_argument('--gpu', type=int, default=2, help='gpu index')
    parser.add_argument('--config', type=str, default='scripts/configs/deform.yaml', help='config file')
    parser.add_argument('--continue_train', action='store_true', help='continue training from checkpoint')
    parser.add_argument('--use_tensorboard', action='store_true', help='use tensorboard')
    parser.add_argument('--shape_checkpoint', type=str, default='checkpoints/baseshape.pth', help='shape checkpoint path')
    parser.add_argument('--deform_checkpoint', type=str, default='checkpoints/deform.pth', help='deform checkpoint path')
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    # Load config
    CFG = yaml.safe_load(open(args.config, 'r'))
    # load shape & deform codes
    shape_checkpoint = torch.load(args.shape_checkpoint, map_location='cpu')
    shape_codes = shape_checkpoint['latent_shape']['weight']
    deform_checkpoint = torch.load(args.deform_checkpoint, map_location='cpu')
    deform_codes = deform_checkpoint['deform_codes']['weight']
    
    # dataset
    deform_dataset = DeformLeafDataset()
    encoder_dataset = EncoderDataset(
        deform_dataset,
        shape_codes,
        deform_codes,
        device=device
    )

    # Create trainer
    trainer = EncoderTrainer(CFG, device, args, encoder_dataset, resolution=128)
    
    # Train
    trainer.train()

