import torch
import os
from scripts.data.dataset import LbsDataset
from scripts.utils.train_utils import BaseTrainer
import argparse 
import yaml

class DeformTrainer(BaseTrainer):
    def __init__(self, decoder, cfg, device, args):
        super(DeformTrainer, self).__init__(decoder, cfg, device, args)
        self.dataset = LbsDataset()
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=args.batch_size, shuffle=True)
        self.decoder = decoder
        self.cfg = cfg
        self.device = device


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RUN Leaf NPM')
    parser.add_argument('--gpu', type=int, default=1, help='gpu index')
    parser.add_argument('--wandb', type=str, default='base shape', help='run name of wandb')
    parser.add_argument('--mode', type=str, default='train', help='mode of train, shape or texture')
    parser.add_argument('--ckpt_shape', type=str, default='checkpoints/baseshape/sdf/latest.pth', help='checkpoint directory')
    parser.add_argument('--output', type=str, default='shape', help='output directory')
    parser.add_argument('--use_wandb', action='store_true', help='use wandb')
    parser.add_argument('--continue_train', action='store_true', help='continue training from latest checkpoint')
    parser.add_argument('--config', type=str, default='scripts/configs/bashshape.yaml', help='config file')
    
    # setting
    args = parser.parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device(f'cuda:{str(args.gpu)}' if torch.cuda.is_available() else 'cpu')
    CFG = yaml.safe_load(open(args.config, 'r')) 

    pass