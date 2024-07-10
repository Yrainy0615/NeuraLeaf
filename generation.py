from torchvision.utils import save_image, make_grid
import torch
import argparse
import yaml
import os
import sys
import torchvision.transforms as transforms
# sys.path.append('scripts')
from scripts.data.dataset import BaseShapeDataset
from scripts.models.decoder import SDFDecoder
from scripts.utils.utils import latent_to_mask, save_tensor_image
from matplotlib import pyplot as plt
import numpy as np
from submodule.pixel2pixel.models import create_model
from submodule.pixel2pixel.options.test_options import TestOptions
from torchvision.transforms import Resize

def generate_baseshape(decoder, latent, dataset,save_folder,mask_transform,k,save_image=False): 
    """
    shape latent -> 2D SDF -> binary mask 
    """
    idx1 = 120
    idx2= 500
    masks1, mask2 = dataset[idx1]['hint'], dataset[idx2]['hint']
    masks_gt = torch.stack([torch.tensor(masks1), torch.tensor(mask2)], dim=0)
    masks_gt = masks_gt.permute(0, 3, 1, 2)
    # masks_gt = masks_gt[:,0,:,:].unsqueeze(1)
    masks_gt = mask_transform(masks_gt)
    latent_x = latent[idx1]
    latent_y = latent[idx2]
    save_name = os.path.join(save_folder, f'inter_{idx1}_{idx2}.png')
    # linear interpolation
    weights = torch.linspace(0, 1, 2).to(latent_x.device)
    latent = latent_x * (1-weights[:, None]) + latent_y * weights[:, None]
    with torch.no_grad():
        masks = latent_to_mask(latent.to('cuda'), decoder=decoder, k=k,size=256)
        masks = masks.unsqueeze(1)
        masks = masks.repeat(1, 3, 1, 1)
        masks = mask_transform(masks)

        if save_image:
            save_tensor_image(masks, save_name)
            save_tensor_image(masks_gt, os.path.join(save_folder, 'gt_mask.png'))

    return masks, masks_gt

def generate_texture(model_texture,masks:torch.tensor,  save_folder, save_image=False):
    """
    generate texture from binary mask
    """

    texture  = model_texture.netG(masks)
    texture = (texture + 1) / 2
    if save_image:
        save_tensor_image(texture, os.path.join(save_folder, 'texture.png'))
    return texture



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RUN Leaf NPM')
    parser.add_argument('--gpu', type=int, default=2, help='gpu index')
    parser.add_argument('--save_folder', type=str, default='results', help='output directory')
    parser.add_argument('--config', type=str, default='scripts/configs/bashshape.yaml', help='config file')
    
    # setting
    args = parser.parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CFG = yaml.safe_load(open(args.config, 'r')) 
    mask_transform =  transforms.Compose([transforms.Normalize((0.5,), (0.5,),(0.5,)),
                                        transforms.Resize((256, 256)),])
    
    # load dataset
    dataset = BaseShapeDataset(CFG['Training']['data_dir'], CFG['Training']['n_sample'])
    
    # load mask decoder
    mask_decoder = SDFDecoder()
    mask_decoder.eval()
    mask_decoder.to(device)
    checkpoint_baseshape = torch.load('checkpoints/baseshape/latest.pth',map_location='cpu')
    mask_decoder.load_state_dict(checkpoint_baseshape['decoder'])
    if 'k' in checkpoint_baseshape:
        k = checkpoint_baseshape['k']
    else:
        k=1
    latent_shape =checkpoint_baseshape['latent_shape']['weight']
    
    # baseshape generation 
    masks, masks_gt = generate_baseshape(mask_decoder, latent_shape, dataset,'./', mask_transform, k, save_image=True)
    
    # texture generation  
    opt_texture = TestOptions().parse()
    opt_texture.gpu_ids = [args.gpu]
    model_texture = create_model(opt_texture)
    model_texture.setup(opt_texture)
    model_texture.eval()
    texture = generate_texture(model_texture, masks, './', save_image=True)

    
    pass
    
    
    
    
    
    