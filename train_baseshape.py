import torch
import torch.optim as optim
import argparse
import os
import sys
import wandb

class TextureTrainer(object):
    def __init__(self, generator_mask):
        self.generator_mask = generator_mask
        self.latent_shape = torch.nn.Parameter(torch.randn(1, 512, 4, 4))


        
    def load_checkpoint(self):
        pass
        
    def reduce_lr(self, epoch):
        if   self.cfg['lr_decay_interval_lat'] is not None and epoch % self.cfg['lr_decay_interval_lat'] == 0:
            decay_steps = int(epoch/self.cfg['lr_decay_interval_lat'])
            lr = self.cfg['lr_lat'] * self.cfg['lr_decay_factor_lat']**decay_steps
            print('Reducting LR for latent codes to {}'.format(lr))
            for param_group in self.optimizer_cameranet.param_groups:
                param_group["lr"] = lr
            
    
    def save_checkpoint(self, epoch,save_name):
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        if save_name == 'latest':
            path = self.checkpoint_path + '/latest.tar'
            torch.save({'epoch': epoch,
                        'encoder3d_state_dict': self.encoder_3d.state_dict(),
                        'cameranet_state_dict': self.cameranet.state_dict(),
                        'generator_state_dict': self.generator.state_dict(),
                        'optimizer_encoder3d_state_dict': self.optimizer_encoder3d.state_dict(),
                        'optimizer_generator_state_dict': self.optimizer_generator.state_dict(),
                        'optimizer_cameranet_state_dict': self.optimizer_cameranet.state_dict(),},
                       path)
        else:
            path = self.checkpoint_path + '/{}__{}.tar'.format(save_name,epoch)
        if not os.path.exists(path):
             torch.save({'epoch': epoch,
                        'encoder3d_state_dict': self.encoder_3d.state_dict(),
                        'cameranet_state_dict': self.cameranet.state_dict(),
                        'generator_state_dict': self.generator.state_dict(),
                        'optimizer_encoder3d_state_dict': self.optimizer_encoder3d.state_dict(),
                        'optimizer_generator_state_dict': self.optimizer_generator.state_dict(),
                        'optimizer_cameranet_state_dict': self.optimizer_cameranet.state_dict(),},
                       path)


    def train(self, epochs):
        loss = 0
        if args.continue_train:
            start = self.load_checkpoint()
        start =0
        ckp_interval =self.cfg['ckpt_interval']
        ckp_vis = self.cfg['ckpt_vis']
        for epoch in range(start, epochs):
            self.reduce_lr(epoch)
            sum_loss_dict = {k: 0.0 for k in self.cfg['lambdas']}
            sum_loss_dict.update({'loss':0.0})
            for batch in self.trainloader:
                loss_dict, texture_gt, texture_pred = self.train_step(batch, epoch,args)
                loss_values = {key: value.item() if torch.is_tensor(value) else value for key, value in loss_dict.items()}
                if args.use_wandb:
                    wandb.log(loss_values)
                    wandb.log({'texture_gt': wandb.Image(texture_gt),
                               'texture_pred': wandb.Image(texture_pred)})
                for k in loss_dict:
                    sum_loss_dict[k] += loss_dict[k]        
            if epoch % ckp_interval ==0 and epoch >0:
                self.save_checkpoint(epoch, save_name=CFG['training']['save_name'])
            # save as latest
            self.save_checkpoint(epoch, save_name='latest')
            n_train = len(self.trainloader)
            for k in sum_loss_dict.keys():
                sum_loss_dict[k] /= n_train
            print_str = "Epoch:{:5d}".format(epoch)
            for k in sum_loss_dict:
                print_str += " " + k + " {:06.4f}".format(sum_loss_dict[k])
            print(print_str)

    def train_step(self, batch, epoch, args):
        self.optimizer_cameranet.zero_grad()
        self.generator.zero_grad()

        loss_total = 0
        for key in loss_dict.keys():
            loss_total += self.cfg['lambdas'][key] * loss_dict[key]
        loss_total.backward()

if __name__ == '__main__':       
    parser = argparse.ArgumentParser(description='RUN Leaf NPM')
    parser.add_argument('--gpu', type=int, default=0, help='gpu index')
    parser.add_argument('--wandb', type=str, default='texture', help='run name of wandb')
    parser.add_argument('--output', type=str, default='shape', help='output directory')
    parser.add_argument('--use_wandb', action='store_true', help='use wandb')
    parser.add_argument('--continue_train', action='store_true', help='continue training from latest checkpoint')
    
    # setting
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = 'NPLM/scripts/configs/npm_def.yaml'
    CFG = yaml.safe_load(open(config, 'r')) 
    if args.use_wandb:
        wandb.init(project='NPLM', name =args.wandb)
        wandb.config.update(CFG)
    

    
    # load generator
    generator = Generator(resolution=256)
    generator.to(device)
    
    trainer = TextureTrainer(encoder_3d=encoder_3d, encoder_2d=encoder_2d,
                            cameranet=cameranet, trainloader=trainloader,
                            decoder_shape=decoder_shape, decoder_deform=decoder_deform,
                            latent_deform=lat_deform_all, latent_shape=lat_idx_all,
                            generator=generator, 
                            cfg=CFG, device=device)
    trainer.train(10001)
    