import torch
import torch.optim as optim
import argparse
import os
import wandb
import yaml
from scripts.data.dataset import BaseShapeDataset
from torchvision.utils import save_image, make_grid
from scripts.utils.utils import latent_to_sdfimage
from tqdm import tqdm
import math
from scripts.models.decoder import SDFDecoder
from torch.nn import functional as F


class BaseTrainer(object):
    def __init__(self, decoder, cfg, device, args):
        self.decoder = decoder
        self.args = args
        self.mode = args.mode
        assert self.mode in ['shape', 'texture']
        self.cfg = cfg['Training']
        self.device = device
        self.dataset = BaseShapeDataset(self.cfg['data_dir'], self.cfg['n_sample'])
        self.dataloader = self.dataset.get_loader(batch_size=self.cfg['batch_size'], shuffle=True)
        self.latent_shape = torch.nn.Embedding(len(self.dataset), cfg['Generator']['Z_DIM'], max_norm=1, device=device)
        self.optim_decoder = optim.Adam(self.decoder.parameters(), lr=self.cfg['LR_D'], betas=(0.0, 0.999))
        torch.nn.init.normal_(self.latent_shape.weight.data, 0.0, 0.1/math.sqrt(cfg['Generator']['Z_DIM']))
        self.optim_latent = optim.Adam(self.latent_shape.parameters(), lr=self.cfg['LR_LAT'], betas=(0.0, 0.999))
        # self.PROGRESSIVE_EPOCHS= [100] * len(self.cfg['BATCH_SIZES'])

        
    def load_checkpoint(self):
        pass
        
    def reduce_lr(self, epoch):
        if   self.cfg['lr_decay_interval_lat'] is not None and epoch % self.cfg['lr_decay_interval_lat'] == 0:
            decay_steps = int(epoch/self.cfg['lr_decay_interval_lat'])
            lr = self.cfg['LR_LAT'] * self.cfg['lr_decay_factor_lat']**decay_steps
            print('Reducting LR for latent codes to {}'.format(lr))
            for param_group in self.optim_latent.param_groups:
                param_group["lr"] = lr
            
    
    def save_checkpoint(self, checkpoint_path, save_name):
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        save_name = os.path.join(checkpoint_path, save_name)
        torch.save({'decoder': self.decoder.state_dict(),
                    'latent_shape': self.latent_shape.state_dict(),
                    'optim_decoder': self.optim_decoder.state_dict(),
                    'optim_latent': self.optim_latent.state_dict()}, save_name)

    def generate_examples(self , epoch,n=5):
        self.decoder.eval()
        random_index = torch.randint(0, len(self.dataset), (n,), device=self.device)
        random_latent = self.latent_shape(random_index)
        sdf_images = latent_to_sdfimage(random_latent, decoder=self.decoder, size=512)
        grid = make_grid(sdf_images, nrow=1)
        save_folder = f"{self.cfg['save_result']}/"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_image(grid.unsqueeze(1), os.path.join(save_folder, f'sample_{epoch}.png'))
        self.decoder.train()

    def train_step_texture(self, batch):
        pass

    def train(self):
        loss = 0
        if self.args.continue_train:
            start = self.load_checkpoint()
        else:
            start =0
        ckpt_interval =self.cfg['ckpt_interval']
        vis_interval = self.cfg['vis_interval']
        for epoch in range(start,self.cfg['num_epochs']):
            self.reduce_lr(epoch)
            sum_loss_dict = {k: 0.0 for k in self.cfg['lambdas']}
            sum_loss_dict.update({'loss':0.0})
            for batch in self.dataloader:
                if self.mode == 'shape':
                    loss_dict = self.train_step_shape(batch)
                else:
                    loss_dict = self.train_step_texture(batch)
                loss_values = {key: value.item() if torch.is_tensor(value) else value for key, value in loss_dict.items()}    
            if self.args.use_wandb:
                wandb.log(loss_values)
            for k in loss_dict:
                sum_loss_dict[k] += loss_dict[k]
            if epoch % vis_interval == 0:
                self.generate_examples(epoch, n=20)
            if epoch % ckpt_interval == 0 and epoch > 0:
                self.save_checkpoint(checkpoint_path=self.cfg['checkpoint_path'], save_name=f'epoch_{epoch}.pth')
            # print result
            n_train = len(self.dataloader)
            for k in sum_loss_dict:
                sum_loss_dict[k] /= n_train
            printstr = "Epoch: {} ".format(epoch)
            for k in sum_loss_dict:
                printstr += "{}: {:.4f} ".format(k, sum_loss_dict[k])
            print(printstr)
            
        self.save_checkpoint(checkpoint_path=self.cfg['checkpoint_path'], save_name='latest.pth')
            
            
    def train_step_shape(self, batch):
        self.optim_decoder.zero_grad()
        self.optim_latent.zero_grad()
        batch_cuda = {k: v.to(device) for (k, v) in zip(batch.keys(), batch.values()) if torch.is_tensor(v)}
        sdf_gt = batch_cuda['sdf']
        points = batch_cuda['points']
        label = batch_cuda['label']
        idx = batch_cuda['idx']
        latent_shape = self.latent_shape(idx)
        glob_cond = torch.cat([latent_shape.unsqueeze(1).expand(-1, points.shape[1], -1), points], dim=2)

        # train decoder
        sdf_pred = self.decoder(glob_cond)
        loss_mse = F.mse_loss(sdf_pred.squeeze(), sdf_gt)
        lat_reg = latent_shape.norm(2, dim=1).mean()
        loss_dict = {'loss_mse': loss_mse, 'lat_reg': lat_reg}
        
        loss_total = 0
        for key in loss_dict.keys():
            loss_total += loss_dict[key] * self.cfg['lambdas'][key]
        loss_total.backward()
        
        self.optim_decoder.step()
        self.optim_latent.step()       
        loss_dict.update({'loss': loss_total.item()})
        return loss_dict

if __name__ == '__main__':       
    parser = argparse.ArgumentParser(description='RUN Leaf NPM')
    parser.add_argument('--gpu', type=int, default=7
                        , help='gpu index')
    parser.add_argument('--wandb', type=str, default='base shape', help='run name of wandb')
    parser.add_argument('--mode', type=str, default='shape', help='mode of train, shape or texture')
    parser.add_argument('--output', type=str, default='shape', help='output directory')
    parser.add_argument('--use_wandb', action='store_true', help='use wandb')
    parser.add_argument('--continue_train', action='store_true', help='continue training from latest checkpoint')
    parser.add_argument('--config', type=str, default='scripts/configs/bashshape.yaml', help='config file')
    
    # setting
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CFG = yaml.safe_load(open(args.config, 'r')) 
    if args.use_wandb:
        wandb.init(project='NeuraLeaf', name =args.wandb)
        wandb.config.update(CFG)
    
    # load decoder
    decoder = SDFDecoder()
    decoder.train()
    decoder.to(device)
    
  
    # load data
    
    # load trainer
    trainer = BaseTrainer(decoder, CFG, device,args)
    trainer.train()
    
    

    