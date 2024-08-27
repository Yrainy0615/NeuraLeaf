import os
import torch
import torch.optim as optim
import math
import wandb

class BaseTrainer(object):
    def __init__(self, decoder, cfg, device, args):
        self.decoder = decoder
        self.args = args
        self.mode = args.mode
        assert self.mode in ['train', 'eval']
        self.cfg = cfg['Training']
        self.device = device
        self.dataset = None
        self.dataloader = None

        

        
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
                    'optim_latent': self.optim_latent.state_dict(),
                    'k':self.k.data}, save_name)



    def train_step_texture(self, batch):
        pass

    def train(self):
        pass
            
            
    def train_step_shape(self, batch):
        pass
    
    def eval(self, decoder,latent_shape):
        pass