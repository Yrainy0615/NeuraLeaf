import torch
import os
from scripts.data.dataset import LbsDataset
from scripts.utils.train_utils import BaseTrainer
import argparse 
import yaml
from scripts.models.decoder import BoneDecoder, Embedding, BoneRotationPredictor, BoneSWPredictor, UDFNetwork
from scripts.models.encoder import ShapeEncoder 
from scripts.models.neuralbs import NBS
import math
import torch.optim as optim
from pytorch3d.loss import chamfer_distance
import warnings
warnings.filterwarnings("ignore", message="No mtl file provided")
import trimesh

class DeformTrainer(BaseTrainer):
    def __init__(self, encoder, deform_decoder,dataset,cfg, device, args):
        super(DeformTrainer, self)
        self.dataset = LbsDataset()
        self.cfg = cfg['Trainer']
        self.dataloader = self.dataset.get_loader(batch_size=self.cfg['batch_size'], shuffle=True)
        self.encoder = encoder.to(device)
        self.nbs_model = NBS(cfg['NBS'])
        self.args = args
        self.pe = Embedding(in_channels=3)
        self.device = device
        self.num_bones = cfg['NBS']['num_bones']
        self.mlp_rts = BoneRotationPredictor(num_bones=cfg['NBS']['num_bones']).to(device)
        self.deform_decoder = deform_decoder.to(device)
        # self.mlp_dskin = BoneSWPredictor(num_bones=cfg['NBS']['num_bones']).to(device)
        self.mlp_bone = BoneDecoder(num_bones=cfg['NBS']['num_bones']).to(device)
        self.dataset = dataset
        self.skin_aux_predictor = torch.nn.Linear(128, 2).to(device)
        self.latent_lbs = torch.nn.Embedding(len(self.dataset), cfg['NBS']['lat_dim'], max_norm=1, device=device)
        torch.nn.init.normal_(self.latent_lbs.weight.data, 0.0, 0.1/math.sqrt(cfg['NBS']['lat_dim']))
        self.latent_def = torch.nn.Embedding(len(self.dataset), cfg['Deform']['Z_DIM'], max_norm=1, device=device)
        torch.nn.init.normal_(self.latent_def.weight.data, 0.0, 0.1/math.sqrt(cfg['Deform']['Z_DIM']))
        self.optimizer_encoder = optim.Adam(self.encoder.parameters(), lr=self.cfg['LR_E'])
        self.optimizer_latent = optim.Adam(self.latent_lbs.parameters(), lr=self.cfg['LR_LAT'])
        self.optimizer = optim.Adam(list(self.mlp_rts.parameters())  +
                                    list(self.mlp_bone.parameters()) + list(self.skin_aux_predictor.parameters()), lr=self.cfg['LR_D'])
        self.optimizer_latent_def = optim.Adam(self.latent_def.parameters(), lr=self.cfg['LR_LAT'])

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
                loss_dict = self.train_step(batch, epoch)

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
    
    def train_step(self, batch,epoch):
        # zero grad
        self.optimizer.zero_grad()
        self.optimizer_encoder.zero_grad()
        self.optimizer_latent.zero_grad()
        
        # read data
        canonical_points = batch['canonical_points'].to(self.device)
        deformed_points = batch['deform_points'].to(self.device)
        if epoch ==0:
            canonical_mesh = trimesh.PointCloud(canonical_points[0].detach().cpu().numpy())
            canonical_mesh.export(f'results/deform/train/canonical_{epoch}.ply')
            deformed_mesh = trimesh.PointCloud(deformed_points[0].detach().cpu().numpy())
            deformed_mesh.export(f'results/deform/train/deformed_{epoch}.ply')
            
        occ_canonical = batch['occ_canonical'].to(self.device)
        idx = batch['idx'].to(self.device)
        latent_lbs = self.latent_lbs(idx)
        latent_def = self.latent_def(idx)
        # input feature
        feat_input = self.encoder(occ_canonical.float()) # [bs,lat_di]
        feat_input = feat_input.squeeze(1) # [bs, lat_di]
        # print(f'feat:{feat_input[0]}')
        
        # pred bone [bs,B,10]
        bones = self.mlp_bone(feat_input)
        bone_emb = self.pe(bones[..., :3]) # [bs,B,63]
        bones_center = bones[..., :3]
        bones_center_prior = self.nbs_model.bone_prior(canonical_points,self.num_bones).to(self.device)
        chamfer_bone = 0 
        if epoch<50: 
            chamfer_bone = chamfer_distance(bones_center, bones_center_prior)[0]
            

        # pred rts [bs,B,3,4]
        rts_fw = self.mlp_rts(latent_lbs, bone_emb) 
        # pred skin weights [bs,N,B]
        # pts_emb = self.pe(canonical_points) # [bs,N,63]
        # dskin = self.mlp_dskin(latent_lbs,pts_emb) # [bs,N,B]
        dskin = None
        # lbs forward

        skin_aux = self.skin_aux_predictor(feat_input).squeeze()
        skin_fw = self.nbs_model.skinning(bones, canonical_points, dskin, skin_aux)
        deformed_pred, bones_dfm = self.nbs_model.lbs(bones, canonical_points,rts_fw, skin_fw, backward=False)
        loss_chamfer = chamfer_distance(deformed_pred, deformed_points)[0]
        lat_reg_lbs = torch.norm(latent_lbs, p=2)
        lat_reg_def = 0
        # displacement field forward
        # if epoch>50:
        #     glob_cond = torch.cat([latent_def.unsqueeze(1).expand(-1, deformed_pred.shape[1], -1), deformed_pred], dim=2)
        #     delta_x = self.deform_decoder(glob_cond)
        #     deformed_pred_fine = deformed_pred + delta_x
        #     lat_reg_def = torch.norm(latent_def, p=2)
        #     loss_chamfer = chamfer_distance(deformed_pred_fine, deformed_points)[0]
        loss_dict = {'chamfer': loss_chamfer, 'lat_reg': lat_reg_lbs, 'chamfer_bone': chamfer_bone, 'lat_reg_def': lat_reg_def}
        loss_total = 0
        for key in loss_dict.keys():
            loss_total += loss_dict[key] * self.cfg['lambdas'][key]
        loss_total.backward()
        if epoch%10==0:
            deformed_pred_mesh = trimesh.PointCloud(deformed_pred[0].detach().cpu().numpy())
            deformed_pred_mesh.export(f'results/deform/train/deformed_pred_{epoch}.ply')
        self.optimizer.step()
        self.optimizer_encoder.step()
        self.optimizer_latent.step()
        self.optimizer_latent_def.step()
        return loss_dict
        
    
    def save_checkpoint(self, checkpoint_path, save_name):
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        save_name = os.path.join(checkpoint_path, save_name)
        torch.save({
                    'encoder': self.encoder.state_dict(),
                    'bone_decoder': self.mlp_bone.state_dict(),
                    'rts_decoder': self.mlp_rts.state_dict(),
                    'skin_aux_decoder': self.skin_aux_predictor.state_dict(),
                    'lat_lbs': self.latent_lbs.state_dict(),
                    }, save_name)

    def reduce_lr(self, epoch):
        if   self.cfg['lr_decay_interval_lat'] is not None and epoch % self.cfg['lr_decay_interval_lat'] == 0:
            decay_steps = int(epoch/self.cfg['lr_decay_interval_lat'])
            lr = self.cfg['LR_LAT'] * self.cfg['lr_decay_factor_lat']**decay_steps
            print('Reducting LR for latent codes to {}'.format(lr))
            for param_group in self.optimizer_latent.param_groups:
                param_group["lr"] = lr
            for param_group in self.optimizer_encoder.param_groups:
                param_group["lr"] = lr
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RUN Leaf NPM')
    parser.add_argument('--gpu', type=int, default=1, help='gpu index')
    parser.add_argument('--wandb', type=str, default='base shape', help='run name of wandb')
    parser.add_argument('--mode', type=str, default='train', help='mode of train, shape or texture')
    parser.add_argument('--ckpt_shape', type=str, default='checkpoints/baseshape/sdf/latest.pth', help='checkpoint directory')
    parser.add_argument('--output', type=str, default='shape', help='output directory')
    parser.add_argument('--use_wandb', action='store_true', help='use wandb')
    parser.add_argument('--continue_train', action='store_true', help='continue training from latest checkpoint')
    parser.add_argument('--config', type=str, default='scripts/configs/deform.yaml', help='config file')
    
    # setting
    args = parser.parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device(f'cuda:{str(args.gpu)}' if torch.cuda.is_available() else 'cpu')
    CFG = yaml.safe_load(open(args.config, 'r')) 
    
    # load model & dataset
    encoder = ShapeEncoder(output_dim=CFG['NBS']['lat_dim'])
    dataset = LbsDataset()
    deform_decoder = UDFNetwork(d_in=CFG['Deform']['Z_DIM'],
                                d_in_spatial=3,
                         d_hidden=CFG['Deform']['decoder_hidden_dim'],
                         d_out=CFG['Deform']['decoder_out_dim'],
                         n_layers=CFG['Deform']['decoder_nlayers'],
                         udf_type='sdf',
                         geometric_init=False) 
    trainer = DeformTrainer(encoder, deform_decoder,dataset,CFG, device, args)
    trainer.train()
    
    
    pass