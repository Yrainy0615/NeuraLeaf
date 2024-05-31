import torch
import torch.optim as optim
import argparse
import os
import wandb
import yaml
from scripts.models.cylegan import G
from scripts.data.dataset import BaseShapeDataset
from math import log2
from scripts.utils.loss_utils import gradient_penalty
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import math


class BaseTrainer(object):
    def __init__(self, generator, discriminator, cfg, device):
        self.generator = generator
        self.discriminator =  discriminator
        self.cfg = cfg['Training']
        self.device = device
        self.dataset = BaseShapeDataset(self.cfg['data_dir'], None)
        self.latent_shape = torch.nn.Embedding(len(self.dataset), cfg['Generator']['Z_DIM'], max_norm=1, device=device)
        self.optim_gen = optim.Adam(self.generator.parameters(), lr=self.cfg['LR_G'], betas=(0.0, 0.999))
        torch.nn.init.normal_(self.latent_shape.weight.data, 0.0, 0.1/math.sqrt(cfg['Generator']['Z_DIM']))
        self.optim_dis = optim.Adam(self.discriminator.parameters(), lr=self.cfg['LR_D'], betas=(0.0, 0.999))
        self.optim_latent = optim.Adam(self.latent_shape.parameters(), lr=self.cfg['LR_LAT'], betas=(0.0, 0.999))
        self.PROGRESSIVE_EPOCHS= [1000] * len(self.cfg['BATCH_SIZES'])


    def get_loader(self, image_size, cfg, shuffle=True):
        batch_size = cfg['BATCH_SIZES'][int(log2(image_size) / 4)]
        transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()])
        dataset = BaseShapeDataset(cfg['data_dir'], transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return loader
        
    def load_checkpoint(self):
        pass
        
    def reduce_lr(self, epoch):
        if   self.cfg['lr_decay_interval_lat'] is not None and epoch % self.cfg['lr_decay_interval_lat'] == 0:
            decay_steps = int(epoch/self.cfg['lr_decay_interval_lat'])
            lr = self.cfg['LR_LAT'] * self.cfg['lr_decay_factor_lat']**decay_steps
            print('Reducting LR for latent codes to {}'.format(lr))
            for param_group in self.optim_latent.param_groups:
                param_group["lr"] = lr
            
    
    def save_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        save_name = os.path.join(checkpoint_path, 'latest.pth')
        torch.save({'generator': self.generator.state_dict(),
                    'discriminator': self.discriminator.state_dict(),
                    'latent_shape': self.latent_shape.state_dict(),
                    'optim_gen': self.optim_gen.state_dict(),
                    'optim_dis': self.optim_dis.state_dict(),
                    'optim_latent': self.optim_latent.state_dict()}, save_name)

    def generate_examples(self ,steps, n=20):
        self.generator.eval()
        alpha = 1.0
        random_index = torch.randint(0, len(self.dataset), (n,), device=self.device)
        random_latent = self.latent_shape(random_index)
        with torch.no_grad():     
            img = self.generator(random_latent, alpha, steps)
            # make grid and save
            grid = make_grid(img, nrow=4)
            save_folder = f"{self.cfg['save_result']}/step{steps}"
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            save_image(grid, os.path.join(save_folder, 'sample.png'))
        self.generator.train()

    def train(self):
        loss = 0
        if args.continue_train:
            start = self.load_checkpoint()
        start =0
        ckp_interval =self.cfg['ckpt_interval']
        step = int(log2(self.cfg['START_TRAIN_AT_IMG_SIZE']/4))
        
        for num_epochs in self.PROGRESSIVE_EPOCHS[step-1:]:
            alpha = 1e-5
            print(f"Current image size: {4 * 2 ** step}")

            dataloader = self.get_loader(4*2**step, self.cfg, shuffle=True)
            for epoch in range(num_epochs):
                print(f"Epoch [{epoch+1}/{num_epochs}]")
                alpha = self.train_step(dataloader, step, alpha)
            self.generate_examples(step, n=20)
            step +=1
            self.save_checkpoint(checkpoint_path=self.cfg['checkpoint_path'])
            
            
    def train_step(self, dataloader, step, alpha):
        loop = tqdm(dataloader)

        # load data
        for i,batch in enumerate(loop):
            self.optim_dis.zero_grad()
            self.optim_gen.zero_grad()
            self.optim_latent.zero_grad()
            batch_cuda = {k: v.to(device) for (k, v) in zip(batch.keys(), batch.values())}
            real = batch_cuda['mask']
            label = batch_cuda['label']
            idx = batch_cuda['idx']
            latent_shape = self.latent_shape(idx)
            cur_batch_size = real.shape[0]
            # train discriminator
            fake = self.generator(latent_shape, alpha, step)
            loss_dis_real = self.discriminator(real, alpha, step)
            loss_dis_fake = self.discriminator(fake.detach(), alpha, step)
            gp = gradient_penalty(self.discriminator, real, fake, alpha, step, device=self.device)
            loss_D = (-(torch.mean(loss_dis_real) - torch.mean(loss_dis_fake)) * self.cfg['lambdas']['loss_D']
                        + self.cfg['lambdas']['dis_gp'] * gp + self.cfg['lambdas']['dis_drift'] * torch.mean(loss_dis_real ** 2))
            loss_D.backward()
            self.optim_dis.step()
            
            # train generator
            gen_fake = self.discriminator(fake, alpha, step)
            loss_mse = torch.nn.MSELoss()(fake, real)
            lat_reg = (torch.norm(latent_shape, dim=-1) ** 2 ).mean()           
            loss_G = -torch.mean(gen_fake) * self.cfg['lambdas']['loss_G']   +   lat_reg * self.cfg['lambdas']['lat_reg'] + loss_mse * self.cfg['lambdas']['mse']
            loss_G.backward()
            self.optim_gen.step()
            self.optim_latent.step()      
            
            # Update alpha and ensure less than 1
            alpha += cur_batch_size / (
                (self.PROGRESSIVE_EPOCHS[step] * 0.5) * len(self.dataset)
            )
            alpha = min(alpha, 1)

            loop.set_postfix(
                gp=gp.item(),
                loss_G=loss_G.item(),
                loss_D = loss_D.item(),
                latent_reg=lat_reg.item(),
                loss_mse=loss_mse.item(),
            )
            return alpha

if __name__ == '__main__':       
    parser = argparse.ArgumentParser(description='RUN Leaf NPM')
    parser.add_argument('--gpu', type=int, default=7
                        , help='gpu index')
    parser.add_argument('--wandb', type=str, default='base shape', help='run name of wandb')
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
    
    # load generator
    generator = Generator(z_dim=CFG['Generator']['Z_DIM'], w_dim=CFG['Generator']['W_DIM'],
                         in_channels=CFG['Generator']['IN_CHANNELS'],
                         img_channels=CFG['Generator']['IMG_CHANNELS'])
    generator.train()
    generator.to(device)
    
    # load discriminator
    discriminator = Discriminator(in_channels=CFG['Discriminator']['IN_CHANNELS'],
                                  img_channels=CFG['Discriminator']['IMG_CHANNELS'])
    discriminator.train()
    discriminator.to(device)
    
    # load data
    
    # load trainer
    trainer = BaseTrainer(generator, discriminator, CFG, device)
    trainer.train()
    
    

    