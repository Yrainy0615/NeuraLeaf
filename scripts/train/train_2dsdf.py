import os
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import torch
import torch.optim as optim
import argparse
import yaml
from scripts.data.dataset import BaseShapeDataset
from torchvision.utils import save_image, make_grid
from scripts.utils.utils import latent_to_mask
from tqdm import tqdm
import math
from scripts.models.decoder import  UDFNetwork
from scripts.utils.loss_utils import gradient
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


class BaseTrainer(object):
    def __init__(self, decoder, cfg, checkpoint,device, args):
        self.decoder = decoder
        self.args = args
        self.mode = args.mode
        k = checkpoint['k']
        assert self.mode in ['train', 'eval']
        self.cfg = cfg['Training']
        self.device = device
        self.dataset = BaseShapeDataset(self.cfg['data_dir'], self.cfg['n_sample'])
        self.dataloader = self.dataset.get_loader(batch_size=self.cfg['batch_size'], shuffle=True)
        self.latent_shape = torch.nn.Embedding(len(self.dataset), cfg['Base']['Z_DIM'], max_norm=1, device=device)
        # use pretrained shape code

        self.k = torch.nn.Parameter(k).requires_grad_(True)
        self.optim_decoder = optim.Adam(self.decoder.parameters(), lr=self.cfg['LR_D'], betas=(0.0, 0.999))
        # self.optim_k = optim.Adam([self.k], lr=self.cfg['LR_D'], betas=(0.0, 0.999))
        torch.nn.init.normal_(self.latent_shape.weight.data, 0.0, 0.1/math.sqrt(cfg['Base']['Z_DIM']))
        self.optim_latent = optim.Adam(self.latent_shape.parameters(), lr=self.cfg['LR_LAT'], betas=(0.0, 0.999))
        self.optim_k = optim.Adam([self.k], lr=0.001, betas=(0.0, 0.999))
        
        # Initialize TensorBoard writer
        if args.use_tensorboard:
            log_dir = os.path.join('logs', 'tensorboard', f"base_shape_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
            self.global_step = 0
        else:
            self.writer = None
        
    def load_checkpoint(self, file):
        checkpoint_base = torch.load(file)
        self.decoder.load_state_dict(checkpoint_base['decoder'])
        self.decoder.eval()
        self.k = checkpoint_base['k']
        self.k.requires_grad_(False)
        
    def reduce_lr(self, epoch):
        if   self.cfg['lr_decay_interval_lat'] is not None and epoch % self.cfg['lr_decay_interval_lat'] == 0:
            decay_steps = int(epoch/self.cfg['lr_decay_interval_lat'])
            lr = self.cfg['LR_LAT'] * self.cfg['lr_decay_factor_lat']**decay_steps
            print('Reducting LR for latent codes to {}'.format(lr))
            for param_group in self.optim_latent.param_groups:
                param_group["lr"] = lr
            for param_group in self.optim_k.param_groups:
                param_group["lr"] = lr
            for param_group in self.optim_decoder.param_groups:
                param_group["lr"] = lr       
    
    def save_checkpoint(self, checkpoint_path, save_name):
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        save_name = os.path.join(checkpoint_path, save_name)
        torch.save({'decoder': self.decoder.state_dict(),
                    'latent_shape': self.latent_shape.state_dict(),
                    'optim_decoder': self.optim_decoder.state_dict(),
                    'optim_latent': self.optim_latent.state_dict(),
                    'k':self.k.data}, save_name)

    def generate_examples(self , epoch,n=5):
        self.decoder.eval()
        random_index = torch.randint(0, len(self.dataset), (n,), device=self.device)
        random_latent = self.latent_shape(random_index)
        masks = latent_to_mask(random_latent, decoder=self.decoder, size=128, k = self.k)
        grid = make_grid(masks, nrow=2)
        save_folder = f"{self.cfg['save_result']}/"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_image(grid.unsqueeze(1), os.path.join(save_folder, f'train_sample_{epoch}.png'))
        self.decoder.train()

    def train_extra_shape(self, batch):
        self.load_checkpoint('checkpoints/cvpr/epoch_1000_base_deformleaf.pth.pth')
        self.optim_latent.zero_grad()
        batch_cuda = {k: v.to(self.device) for (k, v) in zip(batch.keys(), batch.values()) if torch.is_tensor(v)}
        sdf_gt = batch_cuda['sdf']
        mask_gt = batch_cuda['hint']
        points = batch_cuda['points']
        idx = batch_cuda['idx']
        latent_shape = self.latent_shape(idx)
        glob_cond = torch.cat([latent_shape.unsqueeze(1).expand(-1, points.shape[1], -1), points], dim=2)

        # train decoder
        sdf_pred = self.decoder(glob_cond)
        loss_mse = F.mse_loss(sdf_pred.squeeze(), sdf_gt)
        lat_reg = latent_shape.norm(2, dim=1).mean()
        mask_pred  = latent_to_mask(latent_shape, decoder=self.decoder, size=128, k = self.k)
        loss_mask = F.mse_loss(mask_pred, mask_gt)
        loss_dict = {'loss_mse': loss_mse, 'lat_reg': lat_reg, 'loss_mask': loss_mask}
        
        loss_total = 0
        for key in loss_dict.keys():
            loss_total += loss_dict[key] * self.cfg['lambdas'][key]
        loss_total.backward()
        
        self.optim_latent.step() 
        self.optim_k.step()
        loss_dict.update({'loss': loss_total.item()})
        
        # Log to TensorBoard
        if self.writer is not None:
            for key, value in loss_dict.items():
                if torch.is_tensor(value):
                    self.writer.add_scalar(f'Train/{key}', value.item(), self.global_step)
                else:
                    self.writer.add_scalar(f'Train/{key}', value, self.global_step)
            
            # Log masks (predicted and ground truth)
            if mask_pred.dim() == 2:
                mask_pred = mask_pred.unsqueeze(0)
            if mask_gt.dim() == 2:
                mask_gt = mask_gt.unsqueeze(0)
            if mask_pred.dim() == 3:
                mask_pred = mask_pred.unsqueeze(1)  # Add channel dimension
            if mask_gt.dim() == 3:
                mask_gt = mask_gt.unsqueeze(1)  # Add channel dimension
            
            # Log mask images
            mask_pred_grid = make_grid(mask_pred, nrow=min(4, mask_pred.size(0)), normalize=True)
            mask_gt_grid = make_grid(mask_gt, nrow=min(4, mask_gt.size(0)), normalize=True)
            self.writer.add_image('Train/Mask_Predicted', mask_pred_grid, self.global_step)
            self.writer.add_image('Train/Mask_GroundTruth', mask_gt_grid, self.global_step)
            
            self.global_step += 1
        return loss_dict, mask_gt

    def train(self):
        loss = 0
        if self.args.continue_train:
            start = self.load_checkpoint()
        else:
            start = 0
        ckpt_interval = self.cfg['ckpt_interval']
        save_name = self.cfg['save_name']
        
        for epoch in range(start, self.cfg['num_epochs']):
            self.reduce_lr(epoch)
            sum_loss_dict = {k: 0.0 for k in self.cfg['lambdas']}
            sum_loss_dict.update({'loss': 0.0})
            
            # Use train_step_shape for main training
            for batch in self.dataloader:
                loss_dict = self.train_step_shape(batch)
                for k in loss_dict:
                    if k in sum_loss_dict:
                        if torch.is_tensor(loss_dict[k]):
                            sum_loss_dict[k] += loss_dict[k].item()
                        else:
                            sum_loss_dict[k] += loss_dict[k]
            
            # Log epoch-level metrics
            n_train = len(self.dataloader)
            for k in sum_loss_dict:
                sum_loss_dict[k] /= n_train
                if self.writer is not None:
                    self.writer.add_scalar(f'Epoch/{k}', sum_loss_dict[k], epoch)
            
            # Run evaluation every 1000 epochs
            # if epoch % 1000 == 0 and epoch > 0:
            #     eval_loss_dict = self.eval_step(epoch)
            #     printstr_eval = "Eval Epoch: {} ".format(epoch)
            #     for k in eval_loss_dict:
            #         printstr_eval += "{}: {:.4f} ".format(k, eval_loss_dict[k])
            #     print(printstr_eval)
            
            if epoch % ckpt_interval == 0 and epoch > 0:
                self.save_checkpoint(checkpoint_path=self.cfg['checkpoint_path'], save_name=f'epoch_{epoch}_{save_name}.pth')
            
            # Save latest checkpoint
            self.save_checkpoint(checkpoint_path=self.cfg['checkpoint_path'], save_name=f'latest_{save_name}.pth')
            
            # Print result
            printstr = "Epoch: {} ".format(epoch)
            for k in sum_loss_dict:
                printstr += "{}: {:.4f} ".format(k, sum_loss_dict[k])
            print(printstr)
        
        # Close TensorBoard writer
        if self.writer is not None:
            self.writer.close()      
            
    def train_step_shape(self, batch):
        self.optim_decoder.zero_grad()
        self.optim_latent.zero_grad()
        self.optim_k.zero_grad()
        batch_cuda = {k: v.to(self.device) for (k, v) in zip(batch.keys(), batch.values()) if torch.is_tensor(v)}
        sdf_gt = batch_cuda['sdf']
        mask_gt = batch_cuda['hint']
        points = batch_cuda['points'].clone().detach().requires_grad_() 
        idx = batch_cuda['idx']
        latent_shape = self.latent_shape(idx)
        glob_cond = torch.cat([latent_shape.unsqueeze(1).expand(-1, points.shape[1], -1), points], dim=2)

        # train decoder
        sdf_pred = self.decoder(glob_cond)
        loss_mse = F.mse_loss(sdf_pred.squeeze(-1), sdf_gt)  # l_sdf
        lat_reg = latent_shape.norm(2, dim=1).mean()  # l_reg
        mask_pred  = latent_to_mask(latent_shape, decoder=self.decoder, size=128, k = self.k)
        loss_mask = torch.abs(mask_pred - mask_gt).mean()  # l_sil
        gradients_surf = gradient(sdf_pred, points)
        loss_eikonal  = torch.mean(torch.abs(gradients_surf.norm(dim=-1) - 1))
        loss_dict = {'loss_mse': loss_mse, 'lat_reg': lat_reg, 'loss_mask': loss_mask, 'loss_eikonal': loss_eikonal}  # loss_dict
        
        loss_total = 0
        for key in loss_dict.keys():
            # Get lambda weight, default to 0 if not specified
            lambda_weight = self.cfg['lambdas'].get(key, 0.0)
            loss_total += loss_dict[key] * lambda_weight
        loss_total.backward()
        
        self.optim_decoder.step()
        self.optim_latent.step() 
        self.optim_k.step()     
        loss_dict.update({'loss': loss_total.item()})
        
        # Log to TensorBoard
        if self.writer is not None:
            for key, value in loss_dict.items():
                if torch.is_tensor(value):
                    self.writer.add_scalar(f'Train/{key}', value.item(), self.global_step)
                else:
                    self.writer.add_scalar(f'Train/{key}', value, self.global_step)
            
            # Log masks (predicted and ground truth)
            if mask_pred.dim() == 2:
                mask_pred = mask_pred.unsqueeze(0)
            if mask_gt.dim() == 3:
                mask_gt = mask_gt[:, 0, :, :]  # Take first channel if needed
            if mask_gt.dim() == 2:
                mask_gt = mask_gt.unsqueeze(0)
            if mask_pred.dim() == 3:
                mask_pred = mask_pred.unsqueeze(1)  # Add channel dimension
            if mask_gt.dim() == 3:
                mask_gt = mask_gt.unsqueeze(1)  # Add channel dimension
            
            # Log mask images
            mask_pred_grid = make_grid(mask_pred, nrow=min(4, mask_pred.size(0)), normalize=True)
            mask_gt_grid = make_grid(mask_gt, nrow=min(4, mask_gt.size(0)), normalize=True)
            self.writer.add_image('Train/Mask_Predicted', mask_pred_grid, self.global_step)
            self.writer.add_image('Train/Mask_GroundTruth', mask_gt_grid, self.global_step)
            
            self.global_step += 1
        return loss_dict
    
    def eval_step(self, epoch):
        """Run evaluation on validation/test set during training"""
        self.decoder.eval()
        eval_loss_dict = {k: 0.0 for k in self.cfg['lambdas']}
        eval_loss_dict.update({'loss': 0.0})
        
        # Initialize mask collections for visualization
        all_mask_pred = None
        all_mask_gt = None
        
        with torch.no_grad():
            n_eval = 0
            for batch in self.dataloader:
                batch_cuda = {k: v.to(self.device) for (k, v) in zip(batch.keys(), batch.values()) if torch.is_tensor(v)}
                sdf_gt = batch_cuda['sdf']
                mask_gt = batch_cuda['hint']
                points = batch_cuda['points']
                idx = batch_cuda['idx']
                latent_shape = self.latent_shape(idx)
                glob_cond = torch.cat([latent_shape.unsqueeze(1).expand(-1, points.shape[1], -1), points], dim=2)

                # Forward pass
                sdf_pred = self.decoder(glob_cond)
                loss_mse = F.mse_loss(sdf_pred.squeeze(-1), sdf_gt)
                lat_reg = latent_shape.norm(2, dim=1).mean()
                mask_pred = latent_to_mask(latent_shape, decoder=self.decoder, size=128, k=self.k)
                
                # Handle mask_gt dimensions
                if mask_gt.dim() == 4:
                    mask_gt_2d = mask_gt[:, 0, :, :]  # Take first channel
                else:
                    mask_gt_2d = mask_gt
                
                loss_mask = torch.abs(mask_pred - mask_gt_2d).mean()
                
                # Eikonal loss: compute with gradients enabled (temporarily exit no_grad)
                with torch.enable_grad():
                    points_grad = points.clone().detach().requires_grad_(True)
                    latent_expanded = latent_shape.unsqueeze(1).expand(-1, points_grad.shape[1], -1)
                    # decoder.gradient concatenates [xyz, latent] internally and returns gradients w.r.t. that
                    gradients_all = self.decoder.gradient(points_grad, latent_expanded)
                    # Extract gradient w.r.t. points (first 2 dimensions, since gradient concatenates [xyz, latent])
                    if gradients_all.dim() == 4:
                        gradients_xyz = gradients_all[..., :2]  # First 2 dims are xyz gradient
                        gradients_xyz = gradients_xyz.squeeze(1)  # Remove the extra dimension: [B, N, 2]
                    else:
                        gradients_xyz = gradients_all[..., :2]  # First 2 dims are xyz gradient
                    # Eikonal loss: |gradient| should be 1
                    loss_eikonal = torch.abs(gradients_xyz.norm(dim=-1) - 1.0).mean()
                
                loss_dict = {'loss_mse': loss_mse, 'lat_reg': lat_reg, 'loss_mask': loss_mask, 'loss_eikonal': loss_eikonal.item()}
                
                # Collect masks for visualization (collect from first few batches)
                if n_eval < 3 and self.writer is not None:
                    # Prepare masks for visualization
                    mask_pred_viz = mask_pred.clone()
                    mask_gt_viz = mask_gt_2d.clone()
                    if mask_pred_viz.dim() == 2:
                        mask_pred_viz = mask_pred_viz.unsqueeze(0)
                    if mask_gt_viz.dim() == 2:
                        mask_gt_viz = mask_gt_viz.unsqueeze(0)
                    if mask_pred_viz.dim() == 3:
                        mask_pred_viz = mask_pred_viz.unsqueeze(1)
                    if mask_gt_viz.dim() == 3:
                        mask_gt_viz = mask_gt_viz.unsqueeze(1)
                    
                    # Store masks for later visualization
                    if n_eval == 0:
                        all_mask_pred = mask_pred_viz
                        all_mask_gt = mask_gt_viz
                    else:
                        all_mask_pred = torch.cat([all_mask_pred, mask_pred_viz], dim=0)
                        all_mask_gt = torch.cat([all_mask_gt, mask_gt_viz], dim=0)
                
                loss_total = 0
                for key in loss_dict.keys():
                    # Get lambda weight, default to 0 if not specified
                    lambda_weight = self.cfg['lambdas'].get(key, 0.0)
                    loss_total += loss_dict[key] * lambda_weight
                
                # Accumulate losses
                for k in loss_dict:
                    if k in eval_loss_dict:
                        eval_loss_dict[k] += loss_dict[k].item()
                eval_loss_dict['loss'] += loss_total.item()
                n_eval += 1
        
        # Average losses
        for k in eval_loss_dict:
            eval_loss_dict[k] /= n_eval
        
        # Log to TensorBoard
        if self.writer is not None:
            for key, value in eval_loss_dict.items():
                self.writer.add_scalar(f'Eval/{key}', value, epoch)
            
            # Visualize collected masks
            if all_mask_pred is not None and all_mask_gt is not None:
                # Limit to max 16 masks for visualization
                max_masks = min(16, all_mask_pred.size(0))
                mask_pred_viz = all_mask_pred[:max_masks]
                mask_gt_viz = all_mask_gt[:max_masks]
                
                # Create grid and log to tensorboard
                mask_pred_grid = make_grid(mask_pred_viz, nrow=min(4, mask_pred_viz.size(0)), normalize=True)
                mask_gt_grid = make_grid(mask_gt_viz, nrow=min(4, mask_gt_viz.size(0)), normalize=True)
                self.writer.add_image('Eval/Mask_Predicted', mask_pred_grid, epoch)
                self.writer.add_image('Eval/Mask_GroundTruth', mask_gt_grid, epoch)
        
        self.decoder.train()
        return eval_loss_dict
    
    def eval(self):
        """Standalone evaluation function - uses current decoder state"""
        self.decoder.eval()
        for batch in self.dataloader:
            batch_cuda = {k: v.to(self.device) for (k, v) in zip(batch.keys(), batch.values()) if torch.is_tensor(v)}
            sdf_img_gt = batch_cuda.get('sdf_img', None)
            mask_gt = batch_cuda['hint']
            idx = batch_cuda['idx']
            
            # Use current decoder and latent codes
            latent_shape = self.latent_shape(idx)
            mask_pred = latent_to_mask(latent_shape, decoder=self.decoder, size=512, k=self.k)
            
            # Save results
            save_folder = f"{self.cfg['save_result']}/"
            os.makedirs(save_folder, exist_ok=True)
            
            if sdf_img_gt is not None:
                grid_sdf_gt = make_grid(sdf_img_gt.permute(0,3,1,2), nrow=4)
                save_image(grid_sdf_gt, os.path.join(save_folder, 'sdf_gt.png'))
            
            if mask_gt.dim() == 4:
                grid_mask_gt = make_grid(mask_gt.permute(0,3,1,2), nrow=4)
            else:
                grid_mask_gt = make_grid(mask_gt.unsqueeze(1), nrow=4)
            save_image(grid_mask_gt, os.path.join(save_folder, 'mask_gt.png'))
            
            # Save predicted mask
            if mask_pred.dim() == 2:
                mask_pred = mask_pred.unsqueeze(0)
            if mask_pred.dim() == 3:
                mask_pred = mask_pred.unsqueeze(1)
            grid_mask_pred = make_grid(mask_pred, nrow=4, normalize=True)
            save_image(grid_mask_pred, os.path.join(save_folder, 'mask_pred.png'))

            
if __name__ == '__main__':       
    parser = argparse.ArgumentParser(description='RUN Leaf NPM')
    parser.add_argument('--gpu', type=int, default=1, help='gpu index')
    parser.add_argument('--mode', type=str, default='train', help='mode of train, shape or texture')
    parser.add_argument('--ckpt_shape', type=str, default='checkpoints/baseshape/sdf/latest.pth', help='checkpoint directory')
    parser.add_argument('--output', type=str, default='shape', help='output directory')
    parser.add_argument('--continue_train', action='store_true', help='continue training from latest checkpoint')
    parser.add_argument('--config', type=str, default='scripts/configs/bashshape.yaml', help='config file')
    parser.add_argument('--use_tensorboard', action='store_true', help='use tensorboard')
    
    # setting
    args = parser.parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device(f'cuda:{str(args.gpu)}' if torch.cuda.is_available() else 'cpu')
    CFG = yaml.safe_load(open(args.config, 'r'))
    
    # load decoder
    # decoder = SDFDecoder()
    decoder = UDFNetwork(d_in=CFG['Base']['Z_DIM'],
                         d_hidden=CFG['Base']['decoder_hidden_dim'],
                         d_out=CFG['Base']['decoder_out_dim'],
                         n_layers=CFG['Base']['decoder_nlayers'],
                         udf_type='sdf',
                         geometric_init=False) 
    checkpoint = torch.load('checkpoints/cvpr/epoch_1000_base_deformleaf.pth.pth')
    decoder.load_state_dict(checkpoint['decoder'])

    # load trainer
    trainer = BaseTrainer(decoder, CFG, checkpoint,device,args)
    

    decoder.train()
    decoder.to(device)
    trainer.train()

    

    