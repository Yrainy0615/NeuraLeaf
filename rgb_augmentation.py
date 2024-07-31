import torch
from submodule.SITTA.models import AutoGenerator
from submodule.SITTA.utils import get_config, get_arguments, get_singleM_dataloaders
from scripts.data.data_utils import data_processor
import numpy as np
from scripts.data.dataset import LeafRGBDataset
from torchvision.utils import save_image
from scripts.utils.utils import save_tensor_image



def main():
    opts = get_arguments()
    config = get_config(opts.config)
    config_g = config['netG']
    opts.device = torch.device("cpu" if opts.not_cuda else "cuda:7")

    # load net
    netG_A = AutoGenerator(config_g).to(opts.device)
    netG_B = AutoGenerator(config_g).to(opts.device)
    netG_A.eval()
    netG_B.eval()
    netG_checkpoint = torch.load('results/SITTA/netG_Current.pt')
    netG_A.load_state_dict(netG_checkpoint['netG_A'])
    netG_B.load_state_dict(netG_checkpoint['netG_B'])

    # load data
    processor = data_processor('dataset/2D_Datasets')
    dataset = LeafRGBDataset('dataset/2D_Datasets')
    img_loader = dataset.get_loader(16, True)
    style_loader = dataset.get_loader(16, True)
    for it, (imgs, styles) in enumerate(zip(img_loader, style_loader)):
        imgs = imgs.to(opts.device)
        styles = styles.to(opts.device)
        with torch.no_grad():
            s_A, stats_A = netG_A.encode_s(imgs)
            s_B, stats_B = netG_A.encode_s(styles)
            t_A = netG_A.encode_t(imgs)
            t_B = netG_A.encode_t(styles)

            x_ab = netG_B.decode(s_A, t_B, stats_A)
            x_ba = netG_A.decode(s_B, t_A, stats_B)
            s_A_mean = s_A.detach().mean(dim=1, keepdim=True).repeat(1, 3, 1, 1).type(torch.cuda.FloatTensor)
            s_A_mean = torch.nn.functional.interpolate(s_A_mean, size=(imgs.shape[2], imgs.shape[3]), mode='bilinear',  align_corners=False)
            s_B_mean = s_B.detach().mean(dim=1, keepdim=True).repeat(1, 3, 1, 1).type(torch.cuda.FloatTensor)
            s_B_mean = torch.nn.functional.interpolate(s_B_mean, size=(imgs.shape[2], imgs.shape[3]), mode='bilinear',  align_corners=False)
            # save
            save_tensor_image(imgs, f'results/SITTA/imgs_{it}.png')
            save_tensor_image(styles, f'results/SITTA/styles_{it}.png')
            save_tensor_image(x_ab, f'results/SITTA/x_ab_{it}.png')
            save_tensor_image(x_ba, f'results/SITTA/x_ba_{it}.png')
            save_tensor_image(s_A_mean, f'results/SITTA/s_A_mean_{it}.png')
            save_tensor_image(s_B_mean, f'results/SITTA/s_B_mean_{it}.png')
           
    
    
if __name__ == '__main__':
    opts = get_arguments()
    print('Select Func {}'.format(opts.func))
    if opts.func == 'main':
        main()