from torchvision.utils import save_image, make_grid
import torch
import argparse
import yaml
import os
import sys
sys.path.append('scripts')
from models.decoder import SDFDecoder
from utils.utils import latent_to_sdfimage
from data.mask_sdf import sdf_to_grayscale
from matplotlib import pyplot as plt
import numpy as np

def generate_baseshape(decoder, latent, save_folder): 
    idx1 = 120
    idx2= 600
    latent_x = latent[idx1]
    latent_y = latent[idx2]
    save_name = os.path.join(save_folder, 'baseshape', f'inter_{idx1}_{idx2}.png')
    # linear interpolation
    weights = torch.linspace(0, 1, 10).to(latent_x.device)
    latent = latent_x * weights[:, None] + latent_y * (1 - weights[:, None])
    with torch.no_grad():
        sdf_images = latent_to_sdfimage(latent, decoder=decoder, size=512)
        masks = sdf_images <=   0.01
        masks = masks.float()   
        grid_mask = make_grid(masks.unsqueeze(1), nrow=5)
        grid = make_grid(sdf_images.unsqueeze(1), nrow=5)
        mask_name = save_name.replace('.png', '_mask.png')
        sdf_vis = sdf_to_grayscale(sdf_images[9].cpu().numpy())
        plt.imsave('vis.png', sdf_vis)
        # expand to 3 channels np img
        
        save_image(grid_mask, mask_name)
        save_image(grid, save_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RUN Leaf NPM')
    parser.add_argument('--gpu', type=int, default=7
                        , help='gpu index')
    parser.add_argument('--save_folder', type=str, default='results', help='output directory')
    parser.add_argument('--config', type=str, default='scripts/configs/bashshape.yaml', help='config file')
    
    # setting
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CFG = yaml.safe_load(open(args.config, 'r')) 
    
    # load mask decoder
    mask_decoder = SDFDecoder()
    mask_decoder.eval()
    mask_decoder.to(device)
    checkpoint_baseshape = torch.load('checkpoints/baseshape/sdf/epoch_1000.pth',map_location='cpu')
    mask_decoder.load_state_dict(checkpoint_baseshape['decoder'])
    latent_shape =checkpoint_baseshape['latent_shape']['weight']
    
    # generation 
    generate_baseshape(mask_decoder, latent_shape, args.save_folder)
    
    
    