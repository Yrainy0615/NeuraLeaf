import numpy as np
from pytorch3d import transforms
import torch
import torch.nn as nn
import sys
import cv2
import os
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir) 
sys.path.insert(0, parent_dir)
from utils import geom_utils as gutils
import yaml
import trimesh
from pytorch3d.io import load_obj, load_ply, load_objs_as_meshes
from pytorch3d.loss import chamfer_distance, mesh_edge_loss, mesh_laplacian_smoothing
from pytorch3d.transforms import quaternion_to_matrix
from pytorch3d.structures import Meshes
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend



def create_ellipsoid(center, radii, rotation):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    
    # Rotate the points
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i,j], y[i,j], z[i,j]] = np.dot([x[i,j], y[i,j], z[i,j]], rotation) + center
    
    return x, y, z

class NBS():
    def __init__(self, opts):
        """
        bone: ...,B,10  -B gaussian ellipsoids
        pts: bs,N,3  -N points
        skin: bs,N,B -B skinning matrix
        rts: bs,B,3,4   - B ririd transforms, applied to bone coordinates (points attached to bones in world coords)
        dskin: bs,N,B - delta skinning matrix

        """
        self.vein_file = 'dataset/template_vein.png'
        self.latent_lbs = None
        self.mlp_rts = None
        self.num_bones = opts['num_bones']
        self.skin_aux = None
        self.dskin = None
        self.opts = opts
        self.rts_fw = None 
        self.skin_aux = torch.Tensor([0, 2])  
        self.canonical_points = None
        self.bone = None
        self.bone_center = None

        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def intitialize(self, pts):
        # self.generate_bone(pts)
        self.generate_bone_from_vein(self.vein_file)
        self.canonical_points = pts
        self.dskin = torch.zeros(pts.shape[0], pts.shape[1], self.num_bones)
        R_init = torch.eye(3).repeat(self.num_bones,1,1)
        t_init = torch.zeros(self.num_bones,3,1)
        rts_fw = torch.cat([R_init,t_init],-1)
        self.rts_fw = nn.Parameter(rts_fw)
        skin_aux = torch.Tensor([0, 2])  
        self.skin_aux = nn.Parameter(skin_aux)
        self.optimizer = torch.optim.Adam(
            [
                {'params': [self.bone],'lr': 0.1},
                {'params': [self.rts_fw],'lr': 0.01},
                {'params': [self.skin_aux],'lr': 0.01},
                {'params': [self.dskin],'lr': 0.01}
             ]
        )
    @staticmethod     
    def farthest_point_sampling(pts, K):
        """
        farthest point sampling
        pts: Tensor - bs, N, 3
        K: int - number of points to sample
        """
        if len(pts.shape)==2:
            pts = pts.unsqueeze(0)
        bs, N, _ = pts.shape
        centroids = torch.zeros(bs, K, dtype=torch.long).to(pts.device)
        distance = torch.ones(bs, N).to(pts.device) * 1e10
        farthest = torch.randint(0, N, (bs,), dtype=torch.long).to(pts.device)
        batch_indices = torch.arange(bs).to(pts.device)

        for i in range(K):
            centroids[:, i] = farthest
            centroid = pts[batch_indices, farthest, :].view(bs, 1, 3)
            dist = torch.sum((pts - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]
            
        return pts[batch_indices, centroids]

    def generate_bone(self, pts=None,num_bone=1000,vis=False):
        # sample in uv space ,img_size =512*512, uniformly sample 1000 points as center 
        if pts is None:
            U = np.random.uniform(-1, 1, num_bone)
            V = np.random.uniform(-1, 1, num_bone)

            points_uv = np.stack((U, V), axis=-1)  # Shape: (1000, 2)  
            points_3d = np.zeros((num_bone, 3))
            points_3d[:, 2] = 0
            points_3d[:, 0] = points_uv[:, 0] 
            points_3d[:, 1] = points_uv[:, 1]    
            # move points to uv center 
            center = torch.tensor(points_3d - np.mean(points_3d, axis=0))
            orient = torch.Tensor([1,0,0,0]).repeat(num_bone,1)
            scale = torch.ones(num_bone, 3)
            scale = scale * 0.0001
            bone = torch.cat([center.squeeze(), orient, scale], dim=-1)
            return torch.tensor(bone.float())
            
        else:
            if len(pts.shape)==2:
                pts = torch.tensor(pts).unsqueeze(0).float()
            b, N,_ = pts.shape
            center = self.farthest_point_sampling(pts, self.num_bones)
            # center = pts
            orient = torch.Tensor([1,0,0,0]).repeat(self.num_bones,1).to(pts.device)
            scale = torch.ones(self.num_bones, 3).to(pts.device)
            scale = scale * 0.01
            bone = torch.cat([center.squeeze(), orient, scale], dim=-1)
            if vis:
                gutils.vis_points(bone.squeeze())
            self.bone_center = bone[:,:3].unsqueeze(0).requires_grad_(False)
            # concat center and partial bone
            self.bone =nn.Parameter(bone.unsqueeze(0)).requires_grad_(True)
            return bone


        
    def visualize_bone(self, mesh, bone:torch.Tensor=None, save_path=None):
        """
        bone: B * 10  
        mesh: pytorch3d mesh
        """
        center = bone[0,:, :3]
        quat = bone[0,:, 3:7]
        scale = bone[0,:, 7:]
        # quat to rotation matrix
        rot  = quaternion_to_matrix(quat)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for i in range(self.num_bones):
            x, y, z = create_ellipsoid(center[i].detach().cpu().numpy(), scale[i].detach().cpu().numpy(), rot[i].detach().cpu().numpy())
            ax.plot_surface(x, y, z, color='b', alpha=0.4)
        # add mesh to plot
        verts = mesh[0].detach().cpu().numpy()
        faces = mesh[1][0].detach().cpu().numpy()
        ax.plot_trisurf(verts[:,0], verts[:,1], faces, verts[:,2], cmap='viridis', edgecolor='red')
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.savefig("bone.png")  # 保存图像
        plt.close()


    @staticmethod
    def get_neighbors(point, points):
        # get the nearest neighbor of a point 
        dist = np.linalg.norm(points - point, axis=1)
        idx = np.argsort(dist)
        return points[idx[0]]

    def warp_bw(self,pts):
        bone_rst = self.bone
        
        pass
    
    def warp_fw(self):
        """
        canonical shape -> deformed shape
        """
        bone_canonical = self.bone
        self.optimizer.zero_grad()
        # skinning matrix
        skin_fw = self.skinning(bone_canonical, self.canonical_points, dskin=self.dskin, skin_aux=self.skin_aux)
        # lbs 
        deformed_pred, bones_dfm = self.lbs(bone_canonical, self.canonical_points,self.rts_fw.unsqueeze(0), skin_fw, backward=False)

        # if i%10==0:
        #     save_dir = f"results/deform/lbs_test/{self.num_bones}bones"
        #     if not os.path.exists(save_dir):
        #         os.makedirs(save_dir)
        #     deformed_mesh.export(os.path.join(save_dir, f"deformed_{i}_maple.ply"))
        
        
        return deformed_pred, bones_dfm
        
        
    def lbs(self,bones,pts,rts_fw,skin,backward=False):
        """
        bones: bs,B,10       - B gaussian ellipsoids indicating rest bone coordinates
        rts_fw: bs,B,12       - B rigid transforms, applied to the rest bones
        xyz_in: bs,N,3       - N 3d points after transforms in the root coordinates
        """
        B = bones.shape[-2]
        N = pts.shape[-2]
        bs = rts_fw.shape[0]
        bones = bones.view(-1,B,bones.shape[-1])
        xyz_in = pts.view(-1,N,3)
        # rts_fw = rts_fw.view(-1,B,12)# B,12
        # rmat=rts_fw[:,:,:9]
        # rmat=rmat.view(bs,B,3,3)
        # tmat= rts_fw[:,:,9:12]
        # rts_fw = torch.cat([rmat,tmat[...,None]],-1)
        # rts_fw = rts_fw.view(-1,B,3,4)

        if backward:
            # bones_dfm = self.bone_transform(bones, rts_fw) # bone coordinates after deform
            rts_bw = gutils.rts_invert(rts_fw)
            xyz = self.blend_skinning(bones, rts_bw, skin, xyz_in)
            bones_dfm = gutils.bone_transform(bones, rts_bw) # bone coordinates after deform
        else:
            xyz = self.blend_skinning(bones.repeat(bs,1,1),rts= rts_fw, skin=skin,pts=xyz_in)
            # bones_dfm = gutils.bone_transform(bones, rts_fw) # bone coordinates after deform
        return xyz
    
    def blend_skinning(self, bones,skin,rts,pts):       
        """
        bone: bs,B,10   - B gaussian ellipsoids
        rts: bs,B,3,4   - B ririd transforms, applied to bone coordinates
        pts: bs,N,3     - N 3d points
        skin: bs,N,B   - skinning matrix
        apply rts to bone coordinates, while computing blending globally
        """
        chunk=pts.shape[1]
        B = rts.shape[-3]
        N = pts.shape[-2]
        bones = bones.view(-1,B,bones.shape[-1])
        pts = pts.view(-1,N,3)
        rts = rts.view(-1,B,3,4)
        bs = pts.shape[0]

        pts_out = []
        for i in range(0,bs,chunk):
            pts_chunk = self.blend_skinning_chunk(bones[i:i+chunk], rts[i:i+chunk], 
                                            skin[i:i+chunk], pts[i:i+chunk])
            pts_out.append(pts_chunk)
        pts = torch.cat(pts_out,0)
        return pts
    
    def blend_skinning_chunk(self, bone,rts,skin,pts):
        """
        bone: bs,B,10   - B gaussian ellipsoids
        rts: bs,B,3,4   - B ririd transforms, applied to bone coordinates (points attached to bones in world coords)
        pts: bs,N,3     - N 3d points
        skin: bs,N,B   - skinning matrix
        apply rts to bone coordinates, while computing blending globally
        """
        B = rts.shape[-3]
        N = pts.shape[-2]
        pts = pts.view(-1,N,3)
        rts = rts.view(-1,B,3,4)
        Rmat = rts[:,:,:3,:3] # bs, B, 3,3
        Tmat = rts[:,:,:3,3]
        device = Tmat.device
        
        Rmat_w = (skin[...,None,None] * Rmat[:,None]).sum(2) # bs,N,B,3
        Tmat_w = (skin[...,None] * Tmat[:,None]).sum(2) # bs,N,B,3
        pts = Rmat_w.matmul(pts[...,None]) + Tmat_w[...,None] 
        pts = pts[...,0]
        return pts
            
    def skinning(self, bones, pts, dskin=None, skin_aux=None):
        """
        bone: ...,B,10  - B gaussian ellipsoids
        pts: bs,N,3    - N 3d points
        skin: bs,N,B   - skinning matrix
        """
        chunk=pts.shape[1]
        bs,N,_ = pts.shape
        B = bones.shape[-2]
        if bones.dim()==2: bones = bones[None].repeat(bs,1,1)
        bones = bones.view(-1,B,10)

        skin = []
        for i in range(0,bs,chunk):
            if dskin is None:
                dskin_chunk = None
            else: 
                dskin_chunk = dskin[i:i+chunk]
            skin_chunk = self.skinning_chunk(bones[i:i+chunk], pts[:,i:i+chunk], \
                                dskin=dskin_chunk, skin_aux=skin_aux)
            skin.append( skin_chunk )
        skin = torch.cat(skin,0)
        return skin

    def skinning_chunk(self, bones, pts, dskin=None, skin_aux=None):
        """
        bone: bs,B,10  - B gaussian ellipsoids
        pts: bs,N,3    - N 3d points, usually N=num points per ray, b~=2034
        skin: bs,N,B   - skinning matrix
        """
        device = pts.device
        log_scale= skin_aux[0]
        w_const  = skin_aux[1]
        bs,N,_ = pts.shape
        B = bones.shape[-2]
        if bones.dim()==2: bones = bones[None].repeat(bs,1,1)
        bones = bones.view(-1,B,10)
    
        center, orient, scale = gutils.vec_to_sim3(bones) 
        orient = orient.permute(0,1,3,2) # transpose R

        # mahalanobis distance [(p-v)^TR^T]S[R(p-v)]
        # transform a vector to the local coordinate
        mdis = center.view(bs,1,B,3) - pts.view(bs,N,1,3) # bs,N,B,3
        mdis = gutils.axis_rotate(orient.view(bs,1,B,3,3), mdis[...,None])
        mdis = mdis[...,0]
        mdis = scale.view(bs,1,B,3) * mdis.pow(2)
        # log_scale (being optimized) controls temporature of the skinning weight softmax 
        # multiply 1000 to make the weights more concentrated initially
        inv_temperature = 1000 * log_scale.exp()
        mdis = (-inv_temperature * mdis.sum(3)) # bs,N,B

        if dskin is not None:
            mdis = mdis+dskin

        skin = mdis.softmax(2)
        return skin
    
    def optimize_single(self,epoch,target, base_face):
        for i in range(epoch):
            deformed_pred, bones_dfm = self.warp_fw()
            deformed_mesh_torch = Meshes(verts=deformed_pred, faces=base_face)
            loss_chamfer = chamfer_distance(deformed_pred, target)[0]
            loss_edge = mesh_edge_loss(deformed_mesh_torch)
            loss_smooth = mesh_laplacian_smoothing(deformed_mesh_torch)
            loss = loss_chamfer + loss_edge + loss_smooth
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            with torch.no_grad():  
                self.bone[:,:,:3] = self.bone_center
            print(f"Epoch {i}, Loss: {loss.item()}")
            if i%20==0:
                deformed_cloud = trimesh.Trimesh(deformed_pred[0].detach().cpu().numpy(), base_face[0].detach().cpu().numpy())
                deformed_cloud.export(f"results/deform/lbs/lbs_test/deformed_{i}_leaf2.ply")
            # update learning rate
            if i%100==0 and i!=0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.5
        self.visualize_bone(self.canonical_mesh, self.bone, save_path="results/deform/lbs/lbs_test/bone.png")


if __name__ == '__main__':
    # load data
    canonical_file = "dataset/template.obj"
    deformed_file = "dataset/deformation_cvpr_new/deform_train/leaf_2_deform.ply"
    vein = 'dataset/2D_Datasets/leaf_vein_new/vein_train/C_1_1_3_bot.png'
    opts_file = "scripts/configs/deform.yaml"
    cfg = yaml.load(open(opts_file, 'r'), Loader=yaml.FullLoader)
    # canonical_mesh = load_obj(canonical_file)
    # deformed_mesh = load_obj(deformed_file)
    # canonical_points = canonical_mesh[0]
    # deformed_points = deformed_mesh[0]
    
    # load model
    nbs_model = NBS(cfg['NBS'])
    canonical_mesh = load_obj(canonical_file)
    canonical_points = canonical_mesh[0].unsqueeze(0)
    canonical_face = canonical_mesh[1][0].unsqueeze(0)
    deformed_mesh = load_ply(deformed_file)
    deformed_points = deformed_mesh[0].unsqueeze(0)
    nbs_model.intitialize(canonical_points)
    nbs_model.optimize_single(200, deformed_points,canonical_face)

        
