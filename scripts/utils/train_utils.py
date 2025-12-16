import os
import torch
import torch.optim as optim
import math

class BaseTrainer(object):
    def __init__(self, decoder,dataset, cfg, device, args):
        self.decoder = decoder
        self.args = args
        self.cfg = cfg['Training']
        self.device = device

        self.optim = None
        self.dataset = dataset
        

        
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
        loss = 0
        if self.args.continue_train:
            start = self.load_checkpoint()
        else:
            start =0
        ckpt_interval =self.cfg['ckpt_interval']
        save_name = self.cfg['save_name']
        for epoch in range(start,self.cfg['num_epochs']):
            self.reduce_lr(epoch)
            sum_loss_dict = {k: 0.0 for k in self.cfg['lambdas']}
            sum_loss_dict.update({'loss':0.0})
            for batch in self.dataloader:

                loss_dict, mask_gt = self.train_step_shape(batch)

                loss_values = {key: value.item() if torch.is_tensor(value) else value for key, value in loss_dict.items()}    
           
            for k in loss_dict:
                sum_loss_dict[k] += loss_dict[k]
           
            if epoch % ckpt_interval == 0 and epoch > 0:
                self.save_checkpoint(checkpoint_path=self.cfg['checkpoint_path'], save_name=f'epoch_{epoch}_{save_name}.pth')
            # print result
            n_train = len(self.dataloader)
            for k in sum_loss_dict:
                sum_loss_dict[k] /= n_train
            printstr = "Epoch: {} ".format(epoch)
            for k in sum_loss_dict:
                printstr += "{}: {:.4f} ".format(k, sum_loss_dict[k])
            print(printstr)
            
        self.save_checkpoint(checkpoint_path=self.cfg['checkpoint_path'], save_name='latest_sigmoid.pth')
            
            
    def train_step_shape(self, batch):
        pass
    
    def eval(self, decoder,latent_shape):
        pass