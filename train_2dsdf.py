import torch
import torch.optim as optim
import argparse
import os
import yaml
from scripts.data.dataset import BaseShapeDataset
from torchvision.utils import save_image, make_grid
from scripts.utils.utils import latent_to_mask
from tqdm import tqdm
import math
from scripts.models.decoder import SDFDecoder, UDFNetwork
from torch.nn import functional as F
from scripts.utils.utils import save_tensor_image
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


class BaseTrainer(object):
    def __init__(self, decoder, cfg, checkpoint,device, args):
        self.decoder = decoder
        self.args = args
        self.mode = args.mode
        shape_code = checkpoint['latent_shape']['weight']
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
        save_name = os.path.join(checkpoint_path, save_name+'.pth')
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
        vis_interval = self.cfg['vis_interval']
        save_name = self.cfg['save_name']
        
        for epoch in range(start, self.cfg['num_epochs']):
            self.reduce_lr(epoch)
            sum_loss_dict = {k: 0.0 for k in self.cfg['lambdas']}
            sum_loss_dict.update({'loss': 0.0})
            
            # Use train_step_shape for main training
            for batch in self.dataloader:
                loss_dict, mask_gt = self.train_step_shape(batch)
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
                self.writer.add_scalar(f'Epoch/{k}', sum_loss_dict[k], epoch)
            
            # Log learning rates
            self.writer.add_scalar('LearningRate/decoder', self.optim_decoder.param_groups[0]['lr'], epoch)
            self.writer.add_scalar('LearningRate/latent', self.optim_latent.param_groups[0]['lr'], epoch)
            self.writer.add_scalar('LearningRate/k', self.optim_k.param_groups[0]['lr'], epoch)
            
            if epoch % vis_interval == 0:
                self.generate_examples(epoch, n=4)
                save_tensor_image(mask_gt, os.path.join(self.cfg['save_result'], f'mask_gt_{epoch}.png'))
                
                # Log generated examples to TensorBoard
                self.decoder.eval()
                with torch.no_grad():
                    random_index = torch.randint(0, len(self.dataset), (4,), device=self.device)
                    random_latent = self.latent_shape(random_index)
                    masks = latent_to_mask(random_latent, decoder=self.decoder, size=128, k=self.k)
                    if masks.dim() == 2:
                        masks = masks.unsqueeze(0)
                    if masks.dim() == 3:
                        masks = masks.unsqueeze(1)
                    mask_grid = make_grid(masks, nrow=2, normalize=True)
                    self.writer.add_image('Epoch/Generated_Masks', mask_grid, epoch)
                self.decoder.train()
            
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
        self.writer.close()      
            
    def train_step_shape(self, batch):
        self.optim_decoder.zero_grad()
        self.optim_latent.zero_grad()
        self.optim_k.zero_grad()
        batch_cuda = {k: v.to(self.device) for (k, v) in zip(batch.keys(), batch.values()) if torch.is_tensor(v)}
        sdf_gt = batch_cuda['sdf']
        mask_gt = batch_cuda['hint']
        points = batch_cuda['points']
        idx = batch_cuda['idx']
        latent_shape = self.latent_shape(idx)
        glob_cond = torch.cat([latent_shape.unsqueeze(1).expand(-1, points.shape[1], -1), points], dim=2)

        # train decoder
        sdf_pred = self.decoder(glob_cond)
        loss_mse = F.mse_loss(sdf_pred.squeeze(-1), sdf_gt)
        lat_reg = latent_shape.norm(2, dim=1).mean()
        mask_pred  = latent_to_mask(latent_shape, decoder=self.decoder, size=128, k = self.k)
        loss_mask = torch.abs(mask_pred - mask_gt[:,0,:,:]).mean()
        loss_dict = {'loss_mse': loss_mse, 'lat_reg': lat_reg, 'loss_mask': loss_mask}
        
        loss_total = 0
        for key in loss_dict.keys():
            loss_total += loss_dict[key] * self.cfg['lambdas'][key]
        loss_total.backward()
        
        self.optim_decoder.step()
        self.optim_latent.step() 
        self.optim_k.step()     
        loss_dict.update({'loss': loss_total.item()})
        
        # Log to TensorBoard
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
        return loss_dict, mask_gt
    
    def eval(self, decoder,latent_shape):
        for batch in self.dataloader:
            batch_cuda = {k: v.to(self.device) for (k, v) in zip(batch.keys(), batch.values()) if torch.is_tensor(v)}
            sdf_img_gt = batch_cuda['sdf_img']
            mask_gt = batch_cuda['hint']
            grid_sdf_gt = make_grid(sdf_img_gt.permute(0,3,1,2), nrow=4)
            grid_mask_gt = make_grid(mask_gt.permute(0,3,1,2), nrow=4)
            save_image(grid_sdf_gt, os.path.join(self.cfg['save_result'], 'sdf_gt.png'))
            save_image(grid_mask_gt, os.path.join(self.cfg['save_result'], 'mask_gt.png'))
            idx = batch_cuda['idx']
            latent_shape = latent_shape[idx]
            sdf = latent_to_mask(latent_shape, decoder=decoder, size=512)   
            save_folder = f"{self.cfg['save_result']}/"
            # sdf_to_grayscale for every image in batch
            sdf_img_pred =[]
            # for i in range(len(sdf)):
            #     sdf_img = sdf_to_grayscale(sdf[i])
            #     sdf_img_pred.append(sdf_img)
            # sdf_img_pred = torch.stack(sdf_img_pred)     
            sdf_img_pred = sdf.unsqueeze(1)
            sdf_img_pred = sdf_img_pred.repeat(1, 3, 1, 1)
            grid_sdf_pred = make_grid(sdf_img_pred, nrow=4)
            save_image(grid_sdf_pred, os.path.join(save_folder, f'sdf_pred.png'))
            masks = sdf <=   0.01
            masks = torch.sigmoid(sdf)
            masks = masks.float()
            grid_mask = make_grid(masks.unsqueeze(1), nrow=4)
            save_image(grid_mask, os.path.join(save_folder, 'mask_pred.png'))

            
if __name__ == '__main__':       
    parser = argparse.ArgumentParser(description='RUN Leaf NPM')
    parser.add_argument('--gpu', type=int, default=1, help='gpu index')
    parser.add_argument('--mode', type=str, default='train', help='mode of train, shape or texture')
    parser.add_argument('--ckpt_shape', type=str, default='checkpoints/baseshape/sdf/latest.pth', help='checkpoint directory')
    parser.add_argument('--output', type=str, default='shape', help='output directory')
    parser.add_argument('--continue_train', action='store_true', help='continue training from latest checkpoint')
    parser.add_argument('--config', type=str, default='scripts/configs/bashshape.yaml', help='config file')
    
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
    
    if args.mode == 'train':
        decoder.train()
        decoder.to(device)
        trainer.train()
    elif args.mode == 'eval':
        decoder.eval()
        decoder.to(device)
        decoder_checkpoint = torch.load(args.ckpt_shape)
        decoder.load_state_dict(decoder_checkpoint['decoder'])
        shape_latent = decoder_checkpoint['latent_shape']['weight']
        trainer.eval(decoder, shape_latent)
    
    

    