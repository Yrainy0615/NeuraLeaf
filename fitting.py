import torch
from scripts.data.dataset import DeformLeafDataset
import argparse
import yaml
import os
from scripts.models.neuralbs import NBS
import torch.nn.functional as F
from pytorch3d.structures import Meshes,Pointclouds
from pytorch3d.loss import chamfer_distance, mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency, point_mesh_edge_distance
from scripts.utils.utils import latent_to_mask
from pytorch3d import transforms
from pytorch3d.io import IO, load_objs_as_meshes, load_ply, load_obj
from scripts.utils.loss_utils import ARAPLoss
import trimesh
import numpy as np 
import meshlib.mrmeshpy as mm
import meshlib.mrmeshnumpy as mn
import cv2
from scripts.models.decoder import UDFNetwork, SWPredictor, TransPredictor
from probreg import cpd
# from scripts.utils.vis_utils import visualize_deform_part, visualize_handle

def rigid_cpd_cuda(basepoints:torch.tensor, deformed_points:torch.tensor,use_cuda=True):
    import cupy as cp
    if use_cuda:
        to_cpu = cp.asnumpy
        cp.cuda.Device(3).use()
        cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
    else: 
        cp = np
        to_cpu = lambda x: x
    source_pt = cp.asarray(basepoints)
    target_pt = cp.asarray(deformed_points)
    if source_pt.shape[0]>10000:
        rabdom_index = np.random.choice(source_pt.shape[0], 10000, replace=False)
        source_pt = source_pt[rabdom_index]
    rcpd = cpd.RigidCPD(target_pt, use_cuda=use_cuda)
    tf_param_rgd, _, _ = rcpd.registration(source_pt)
    target_rgd = tf_param_rgd.transform(target_pt)
    return target_rgd

class Reconstructor():
    def __init__(self,  cfg_deform, device, shape_decoder, sw_predictor, trans_predictor, shape_codes, deform_codes, bone,k):
        self.device = device
        self.dataset = DeformLeafDataset()
        self.cfg_deform = cfg_deform['Trainer']
        self.dataloader = self.dataset.get_loader(batch_size=1)
        self.nbs_model = NBS(cfg_deform['NBS'])
        self.shape_decoder = shape_decoder
        self.sw_predictor = sw_predictor
        self.trans_predictor = trans_predictor
        self.shape_codes = shape_codes # [231,256]
        self.deform_codes = deform_codes # [152,128]
        self.shape_codes = self.shape_codes.to(self.device)
        self.deform_codes = self.deform_codes.to(self.device)
        self.bone = bone
        self.k = k
        self.k.to(self.device)

    
    def mask2mesh(self,masks):
        meshes = []
        for mask in masks:
            mask = mask.squeeze(0).detach().cpu().numpy()
            y_indices, x_indices = np.where(mask == 1)
            z_coords = np.zeros_like(x_indices)
            u_coords = x_indices / mask.shape[1]
            v_coords = (y_indices / mask.shape[0])
            verts = np.stack([u_coords, v_coords, z_coords], axis=-1)
            # random sample 15000 points
            idx = np.random.choice(verts.shape[0], 15000, replace=False)
            verts = verts[idx]
            pc =mn.pointCloudFromPoints(verts)
            
            pc.validPoints = mm.pointUniformSampling(pc, 1e-3)
            pc.invalidateCaches()
            mesh = mm.triangulatePointCloud(pc)
            verts = mn.getNumpyVerts(mesh)
            faces = mn.getNumpyFaces(mesh.topology)
                       
        return torch.tensor(faces).to(self.device) 
    def denformleaf_reconstruction(self, method=None):
        """
        Mesh reconstruction of deform points in DeformLeaf
        """
        extra_base_dir = 'dataset/cvpr_final/base_shape_extra'
        extra_base_files = [os.path.join(extra_base_dir, f) for f in os.listdir(extra_base_dir) if f.endswith('.obj')]
        extra_base_files = sorted(extra_base_files)
        chamfer_total = 0
        for i, data in enumerate(self.dataloader):
            # load data
            base_name = data['base_name']
            deform_name = data['deform_name']
            base_file = data['base_mesh']
            deform_points = data['deform_points']
            deform_points = deform_points.to(self.device)
            if 'deform' in deform_name[0]:
                for base_file in extra_base_files:
                # excute only for maple leaf
                    base_mesh = load_objs_as_meshes([base_file])
                    base_name = base_file.split('/')[-1].split('.')[0]
                    mask_name = base_name.replace('_rgb', '')
                    base_name = base_name + f'_{i}'
                    idx = data['idx']
                    if method == 'direct':
                        chamfer = self.fitting_direct(base_mesh, deform_points,idx,base_name,mask_name)
                    print(f'shape reconstruction of {base_name[0]} is done')
                    chamfer_total += chamfer
        chamfer_mean = chamfer_total/len(self.dataset)
        print(f'mean chamfer distance: {chamfer_mean}')

    def neuraleaf_fitting(self,base_mesh,base_name,points,epoch=300,save_mesh=False):
        for batch in self.dataloader:
            base_name_batch = batch['base_name']
            if base_name == base_name_batch[0]:
                deform_name = batch['deform_name']
                # base_mesh = batch['base_mesh']
                # base_mesh = load_objs_as_meshes(base_mesh)
                # deform_points = batch['deform_points']
                idx = batch['idx']
                shape_idx = batch['shape_idx']
                # forward 

                shape_code = torch.nn.Parameter(self.shape_codes[shape_idx])
                deform_code = torch.nn.Parameter(self.deform_codes[idx])
                optimizer_shape= torch.optim.Adam([shape_code], lr=0.001)
                optimizer_deform = torch.optim.Adam([deform_code], lr=0.001)
                if points is not None:
                    deform_points = points
                    for i in range(epoch):
                    # base_mask = latent_to_mask(shape_code, self.shape_decoder, size=256,k=self.k) # [1,256,256]
                    # base_faces = self.mask2mesh(base_mask)
                    # base_faces = base_faces.unsqueeze(0)
                    # verts_2d = torch.stack([indices[2], indices[1]], dim=-1)/base_mask.shape[1]
                    # verts_2d = verts_2d - verts_2d.mean(0)
                    # # verts_2d to 3d
                    # verts = torch.cat([verts_2d, torch.zeros((verts_2d.shape[0],1)).to(self.device)], dim=-1)
                        base_points = base_mesh.verts_packed().unsqueeze(0)
                        base_points = base_points.to(self.device)
                        # exchange x and z axis
                        # base_points = base_points[:, :, [-2, 1, 0]]
                        base_faces = base_mesh.faces_packed().unsqueeze(0)  
                        base_faces = base_faces.to(self.device)
                        rts_fw = self.trans_predictor(deform_code, self.bone)
                        sw = self.sw_predictor(shape_code, base_points)
                        v0 = self.deform_lbs(self.bone,base_points, sw, rts_fw)
                        new_mesh = Meshes(verts=v0, faces=base_faces,textures=base_mesh.textures)
                        loss_chamfer = chamfer_distance(deform_points.to(self.device).unsqueeze(0), v0)[0]
                        reg_shape = torch.norm(shape_code, dim=-1).mean()
                        reg_deform = torch.norm(deform_code, dim=-1).mean()
                        # loss_edge = mesh_edge_loss(new_mesh)
                        # loss_laplacian = mesh_laplacian_smoothing(new_mesh)
                        loss_dict = {'shape': loss_chamfer, 'reg_shape': reg_shape, 'reg_deform': reg_deform}
                        loss_total = 0
                        for key in loss_dict.keys():
                            loss_total += loss_dict[key]* self.cfg_deform['lambdas'][key]
                        loss_total.backward()
                        print(f'epoch {i} loss: {loss_total}, shape: {loss_chamfer}')
                        optimizer_shape.step()
                        optimizer_deform.step()
                        if i % 300==0:
                            self.reduce_lr(optimizer_shape)
                            self.reduce_lr(optimizer_deform)
        return loss_chamfer, new_mesh
                            
                        

    def raw_to_canonical(self,points):
        """
        Transform raw points to canonical points
        """
        # centering
        points = points - points.mean(0)
        # scaling
        scale = torch.max(torch.abs(points))
        points = points/scale
        return points

    def reduce_lr(self, optimizer, factor=0.9):
        for param_group in optimizer.param_groups:
            param_group['lr'] *= factor

    def find_boundry_verts(self, maskfile, mesh):
        # get bounary of mask
        verts = mesh.verts_packed().squeeze(0)
        verts = verts-verts.mean(0)
        mask = cv2.imread(maskfile, cv2.IMREAD_GRAYSCALE)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boundary_points = contours[0]
        boundary_points = boundary_points.reshape(-1, 2) 
        height, width = mask.shape
        boundary_points_normalized = boundary_points / np.array([width, height])
        boundary_points_normalized -= np.mean(boundary_points_normalized, axis=0)
        boundary_points_tensor = torch.tensor(boundary_points_normalized, dtype=torch.float32)
        # boundary_point_3d = torch.zeros((boundary_points_tensor.shape[0],3)).to(self.device)
        # boundary_point_3d[:, :2] = boundary_points_tensor
        # tri_boundary = trimesh.PointCloud(vertices=boundary_point_3d.cpu().numpy())
        boundary_indices_set = set()  # 使用集合来存储索引，以确保唯一性
        for i in range(boundary_points_tensor.shape[0]):
            boundary_point = boundary_points_tensor[i]
            distances = torch.norm(verts[:,:2] - boundary_point, dim=1)  # 计算所有顶点与该轮廓点的距离
            sorted_indices = torch.argsort(distances)  
            for min_index in sorted_indices:
                if min_index.item() not in boundary_indices_set:
                    boundary_indices_set.add(min_index.item())
                    break
        boundary_indices = torch.tensor(list(boundary_indices_set))
        return boundary_indices
    def get_boundary_edge(self, boundary_indices):
        num_points = boundary_indices.shape[0]
        edges = []
        for i in range(num_points):
            start_idx = boundary_indices[i].item() 
            end_idx = boundary_indices[(i + 1) % num_points].item() 
            edges.append((start_idx, end_idx))
        return edges
    def compute_edge_length(self,verts,edges):
        initial_lengths = []

        for start_idx, end_idx in edges:
            start_point = verts[start_idx]
            end_point = verts[end_idx]
            edge_length = torch.norm(start_point - end_point)
            initial_lengths.append(edge_length)

        return torch.tensor(initial_lengths, dtype=verts.dtype, device=verts.device)


    def fitting_npm(self, base_mesh, deform_points,epoch=601,save_mesh= False):
        """
        Direct optimize skinning weights
        """
        # init skinning weights
        base_points = base_mesh.verts_packed().unsqueeze(0)
        # scale base points to cover deform points 
        # base_points = self.raw_to_canonical(base_points.squeeze())
        scale_diff = torch.abs(torch.max(torch.abs(deform_points)-torch.max(torch.abs(base_points))))
        base_faces = base_mesh.faces_packed().unsqueeze(0)
        # rescaling base points, larger than 
        base_points = base_points*2
        
        deform_points = deform_points.to(self.device)# v0 = self.deform_lbs(self.bone,base_points, sw_softmax, T.unsqueeze(0))

        # base_area = torch.tensor(base_area).to(self.device)
        delta_x = torch.nn.Parameter(torch.zeros_like(base_points).to(self.device))
        optimizer = torch.optim.Adam([delta_x], lr=0.01)

        for i in range(epoch):
            optimizer.zero_grad()
            base_points = base_points.to(self.device)
            base_faces = base_faces.to(self.device)
            # vis
            v0 = base_points + delta_x
            new_mesh = Meshes(verts=v0, faces=base_faces,textures=base_mesh.textures)
            if deform_points.ndimension() == 2:
                deform_points = deform_points.unsqueeze(0)
            loss_chamfer = chamfer_distance(deform_points, v0)[0]            
            loss_edge = mesh_edge_loss(new_mesh)
            loss_laplacian = mesh_laplacian_smoothing(new_mesh)
            loss_dict = {'shape': loss_chamfer, 'edge': loss_edge, 'smooth': loss_laplacian} # 'pme':loss_pme
            loss_total = 0
            for key in loss_dict.keys():
                loss_total += loss_dict[key] * self.cfg_deform['lambdas'][key]
            loss_total.backward()   
            print(f'epoch {i} loss: {loss_total}, shape: {loss_chamfer}, edge: {loss_edge}, smooth: {loss_laplacian}')
            optimizer.step()
            if i % 500==0:
                self.reduce_lr(optimizer)
        return loss_chamfer, new_mesh
  

    def fitting_direct(self, base_mesh, deform_points,epoch=601,save_mesh= False):
        """
        Direct optimize skinning weights
        """
        # init skinning weights
        base_points = base_mesh.verts_packed().unsqueeze(0)
        # scale base points to cover deform points 
        # base_points = self.raw_to_canonical(base_points.squeeze())
        scale_diff = torch.abs(torch.max(torch.abs(deform_points)-torch.max(torch.abs(base_points))))
        base_faces = base_mesh.faces_packed().unsqueeze(0)
        # rescaling base points, larger than 
        base_points = base_points*2
        # base_trimesh = trimesh.Trimesh(vertices=base_points[0].cpu().numpy(), faces=base_faces[0].cpu().numpy())
        # base_area = base_trimesh.area
        # deform_points = self.raw_to_canonical(deform_points)  
        deform_points = deform_points.to(self.device)# v0 = self.deform_lbs(self.bone,base_points, sw_softmax, T.unsqueeze(0))

        # base_area = torch.tensor(base_area).to(self.device)
        b,N,_ = base_points.size()
        if self.bone is None:
            bone_tensor = self.nbs_model.generate_bone() 
            bone_tensor = bone_tensor[:,:3] #[K,3]
            bone = torch.nn.Parameter(bone_tensor.to(self.device))
            self.bone = bone
        K = self.bone.size(0)
        sw = torch.nn.Parameter(torch.rand(N, K).uniform_(0.1, 1.0).to(self.device))        
        t = torch.zeros(K, 7).to(self.device)
        t[:, 3] = 1  
        T = torch.nn.Parameter(t)
        # torch.nn.utils.clip_grad_value_([sw], clip_value=1e-4)
        delta_x = torch.nn.Parameter(torch.zeros_like(base_points).to(self.device))
        optimizer_sw = torch.optim.Adam([sw], lr=0.1)
        optimizer_bone = torch.optim.Adam([self.bone], lr=0.5)
        optimizer_T = torch.optim.Adam([T], lr=0.1)
        for i in range(epoch):
            optimizer_bone.zero_grad()
            optimizer_sw.zero_grad()
            optimizer_T.zero_grad()
            # optimizer_x.zero_grad()
            sw_softmax = F.softmax(sw, dim=1)
            sw_softmax = sw_softmax.unsqueeze(0)
            T = T.to(self.device)
            base_points = base_points.to(self.device)
            base_faces = base_faces.to(self.device)
            # vis
            v0 = self.deform_lbs(self.bone,base_points.unsqueeze(0), sw_softmax, T.unsqueeze(0))
            new_mesh = Meshes(verts=v0, faces=base_faces,textures=base_mesh.textures)
            if deform_points.ndimension() == 2:
                deform_points = deform_points.unsqueeze(0)
            # new_length = self.compute_edge_length(v0.squeeze(0),boundry_edge).to(self.device)
            # loss_length = (new_length - initial_lengths).abs().mean()
            # new_area = self.surface_area(v0, base_faces)
            # loss_area = (new_area - base_area).abs()
            # arap = ARAPLoss(base_points, base_faces)
            # loss_arap = arap(v0, base_points)[0]
            # loss_normal = mesh_normal_consistency(new_mesh)
            loss_chamfer = chamfer_distance(deform_points, v0)[0]            
            loss_edge = mesh_edge_loss(new_mesh)
            loss_laplacian = mesh_laplacian_smoothing(new_mesh)
            loss_dict = {'shape': loss_chamfer, 'edge': loss_edge, 'smooth': loss_laplacian} # 'pme':loss_pme
            loss_total = 0
            for key in loss_dict.keys():
                loss_total += loss_dict[key] * self.cfg_deform['lambdas'][key]
            loss_total.backward()   
            print(f'epoch {i} loss: {loss_total}, shape: {loss_chamfer}, edge: {loss_edge}, smooth: {loss_laplacian}')
            optimizer_sw.step()
            optimizer_bone.step()
            optimizer_T.step()
            # optimizer_x.step()
            if i % 500==0:
                self.reduce_lr(optimizer_sw)
                self.reduce_lr(optimizer_bone)
                self.reduce_lr(optimizer_T)
                # self.reduce_lr(optimizer_x)
        # get max 50 control points  for visualization
        if save_mesh:
            IO().save_mesh(new_mesh,save_name)
            visualize_deform_part(base_points, base_faces, self.bone, sw_softmax, save_name.replace('.obj', '_part.obj'))
        
        return loss_chamfer, new_mesh
    
    def surface_area(self, verts, faces):
        verts = verts.squeeze(0)
        faces = faces.squeeze(0)
        v0 = verts[faces[:, 0]]  # (F, 3)
        v1 = verts[faces[:, 1]]  # (F, 3)
        v2 = verts[faces[:, 2]]  # (F, 3)
        e1 = v1 - v0  # (F, 3)
        e2 = v2 - v0  # (F, 3)        # rescaling base points, larger than 

        cross_product = torch.cross(e1, e2, dim=1)  # (F, 3)
        area = 0.5 * torch.norm(cross_product, dim=1)  # (F,)
        total_area = torch.sum(area)
        return total_area
         
    def deform_lbs(self, bone,base_points, sw, rts_fw):
        """
        rts_fw:(b,K,7)3000
        sw: (b,N,K)
        bone: (B,K,3)
        
        """
        bone = bone.unsqueeze(0).expand(base_points.shape[0], -1, -1).to(self.device)
        B, N, K = sw.shape
        B, K, _ = bone.shape
        v0 = base_points.view(-1,3)
        disp = rts_fw[:,:,:3]
        rot = rts_fw[:,:,3:]
        rot = transforms.quaternion_to_matrix(rot.view(B*K, 4).contiguous()).view(B, K, 3, 3).contiguous()
        hd_disp = torch.repeat_interleave(disp, N, dim=0)
        hd_rot = torch.repeat_interleave(rot, N, dim=0)
        hd_pos = torch.repeat_interleave(bone, N, dim=0)
        per_hd_v = torch.einsum("abcd,abd->abc", hd_rot, (v0[:, None] - hd_pos)) + hd_pos + hd_disp  # (B*V, 40, 3)
        # per_hd_v = torch.einsum("abcd,abd->abc", hd_rot, (v0[:, None]))  + hd_disp  # (B*V, 40, 3)
        region_score = sw.view(-1, K)
        v = torch.sum(region_score[:, :, None] * per_hd_v, 1)  # (B*V, 3)
        return v.view(B, N, 3)
        
    def find_boundary(self, mesh):
        # 获取顶点和面信息
        verts = mesh.verts_packed()  # 顶点 (V, 3)
        faces = mesh.faces_packed()  # 面 (F, 3)

        # 创建一个字典来存储每条边的相邻面数
        edge_count = {}

        for face in faces:
            # 获取每个面的三条边
            edges = [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]

            for edge in edges:
                edge = tuple(sorted(edge))  # 确保每条边的顺序一致
                if edge not in edge_count:
                    edge_count[edge] = 0
                edge_count[edge] += 1

        # 找到只属于一个面的边（即边界边）
        boundary_edges = [edge for edge, count in edge_count.items() if count == 1]

        # 找到边界边的顶点
        boundary_vertices = set()
        for edge in boundary_edges:
            boundary_vertices.update(edge)

        # 将边界顶点转换为张量
        boundary_vertices = torch.tensor(list(boundary_vertices), dtype=torch.long)
        return verts[boundary_vertices]     
    
def denseleaf_reconstruction(dense_leaf_path, reconstrucor, base_mesh, base_name,method='direct',regis_path=None):
    raw_leaf_file = [os.path.join(dense_leaf_path,f) for f in os.listdir(dense_leaf_path) if f.endswith('.ply') or f.endswith('.obj')]        
    for file in raw_leaf_file:
        # check processed file
        ply_name = os.path.basename(file).split('.')[0]
        save_path = os.path.join(dense_leaf_path, 'fit') 
        save_name = os.path.join(save_path, f'{ply_name}_fit.ply')
        save_name_npm = os.path.join(save_path, f'{ply_name}_fit_npm.ply')
        if not os.path.exists(save_name) or not os.path.exists(save_name_npm):
            if file.endswith('.obj'):
                ply_data_raw = load_obj(file)
            elif file.endswith('.ply'):
                ply_data_raw = load_ply(file)
            raw_points = ply_data_raw[0]
            ply_data_raw_canonical = reconstrucor.raw_to_canonical(raw_points)
            base_points = base_mesh.verts_packed().unsqueeze(0)
            base_face = base_mesh.faces_packed().unsqueeze(0)
            # rigid registration
            regis_save_path = dense_leaf_path.replace('split', 'split_regis')
            os.makedirs(regis_save_path,exist_ok=True)
            regis_file = os.path.join(regis_save_path, os.path.basename(file))
            if os.path.exists(regis_file):
                if regis_file.endswith('.obj'):
                    raw_points_regis = load_obj(regis_file)[0]
                elif regis_file.endswith('.ply'):
                    raw_points_regis = load_ply(regis_file)[0]
            else:
                raw_points_regis = rigid_cpd_cuda(base_points.squeeze(), ply_data_raw_canonical)
                raw_points_regis = torch.tensor(raw_points_regis).unsqueeze(0).float().to(base_mesh.device)
                raw_points_regis_save = trimesh.PointCloud(vertices=raw_points_regis.squeeze().cpu().numpy())
                raw_points_regis_save.export(regis_file)
            base_points_save = trimesh.PointCloud(vertices=base_points.squeeze().cpu().numpy())
            # fitting
            if method == 'direct':
                loss_chamfer, new_mesh =reconstrucor.fitting_direct(base_mesh, raw_points_regis,epoch=1001)
                os.makedirs(save_path,exist_ok=True)
                IO().save_mesh(new_mesh, save_name)

                # back to raw
                new_points_regis = rigid_cpd_cuda(raw_points.detach().cpu().squeeze(0).numpy(),new_mesh.verts_packed().squeeze().detach().cpu().numpy())
                new_points_regis = torch.tensor(new_points_regis).unsqueeze(0).float().to(base_mesh.device)

                new_mesh_regis = Meshes(verts=new_points_regis, faces=base_face,textures=base_mesh.textures)
                IO().save_mesh(new_mesh_regis, os.path.join(save_path, f'{ply_name}_regis.obj'))
                print(f'{ply_name} is done') 
            elif method == 'npm':
                loss_chamfer, new_mesh = reconstrucor.fitting_npm(base_mesh,raw_points_regis,epoch=1001)
                IO().save_mesh(new_mesh, save_name_npm)
                 # back to raw
                new_points_regis = rigid_cpd_cuda(raw_points.detach().cpu().squeeze(0).numpy(),new_mesh.verts_packed().squeeze().detach().cpu().numpy())
                new_points_regis = torch.tensor(new_points_regis).unsqueeze(0).float().to(base_mesh.device)

                new_mesh_regis = Meshes(verts=new_points_regis, faces=base_face,textures=base_mesh.textures)
                IO().save_mesh(new_mesh_regis, os.path.join(save_path, f'{ply_name}_regis.obj'))
                print(f'{ply_name} is done')                

            # save mesh

    
        else:
            print(f'{ply_name} is already processed')   
            
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RUN Leaf NPM')
    parser.add_argument('--gpu', type=int, default=3, help='gpu index')
    parser.add_argument('--save_folder', type=str, default='results', help='output directory')
    parser.add_argument('--config', type=str, default='scripts/configs/bashshape.yaml', help='config file')
    
    # setting
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device(f'cuda:{str(args.gpu)}' if torch.cuda.is_available() else 'cpu')
    CFG = yaml.safe_load(open(args.config, 'r')) 
    CFG_deform = yaml.safe_load(open('scripts/configs/deform.yaml', 'r'))
    
    # load shape model
    shape_decoder = UDFNetwork(d_in=CFG['Base']['Z_DIM'],
                         d_hidden=CFG['Base']['decoder_hidden_dim'],
                         d_out=CFG['Base']['decoder_out_dim'],
                         n_layers=CFG['Base']['decoder_nlayers'],
                         udf_type='sdf',
                         geometric_init=False) 
    shape_decoder.eval()
    shape_decoder.to(device)
    checkpoint_base = torch.load('checkpoints/cvpr/epoch_1000_base_deformleaf.pth.pth',map_location='cpu')
    shape_decoder.load_state_dict(checkpoint_base['decoder'])
    k = checkpoint_base['k']
    shape_codes = checkpoint_base['latent_shape']['weight']
    # load deformation model
    num_bones = CFG_deform['NBS']['num_bones']
    swpredictor = SWPredictor(num_bones=num_bones).to(device)
    transpredictor = TransPredictor(num_bones=num_bones).to(device)
    checkpoint_deform = torch.load('checkpoints/cvpr/latest_deform_shape_prior.pth', 'cpu')
    bone = checkpoint_deform['bone']
    bone = bone.to(device)
    deform_codes = checkpoint_deform['deform_codes']['weight']
    deform_codes = deform_codes.to(device)
    swpredictor.load_state_dict(checkpoint_deform['SWpredictor'])
    transpredictor.load_state_dict(checkpoint_deform['Transpredictor'])
    swpredictor.eval()
    transpredictor.eval()
    swpredictor.to(device)
    transpredictor.to(device)
    
    # Reconstructor
    reconstructor = Reconstructor(CFG_deform,device,shape_decoder,swpredictor,transpredictor,shape_codes,deform_codes,bone,k)
    denseleaf_path = 'dataset/denseleaf/data_00001/split'
    base_mesh = load_objs_as_meshes(['dataset/cvpr_final/base_shape/leaf_5.obj'])  # leaf 19 for data2, leaf 5 for data1, 68 for data3, 22 for data4
    base_name = 'leaf_22'
    denseleaf_reconstruction(denseleaf_path, reconstructor, base_mesh, base_name,method='npm',regis_path=None)
    # reconstructor.denformleaf_reconstruction(method= 'direct')
    # points_dir = 'dataset/deformation_cvpr_new/deform_train'
    # ply_files = [os.path.join(points_dir, f) for f in os.listdir(points_dir) if 'deform' in f]
    # for file in ply_files:
    #     ply = load_ply(file)
    #     # ply = load_ply('dataset/deformation_cvpr_new/deform_train/leaf_3_deform.ply')
    #     points = ply[0]
    #     filename = os.path.basename(file).split('.')[0]
    #     points = reconstructor.raw_to_canonical(points)
    #     base_name = filename.replace('_deform', '')
    #     base_mesh = load_objs_as_meshes([f'dataset/cvpr_final/base_shape/{base_name}.obj'])
    #     vis_meshes = 'results/cvpr/rebuttal/npm'
    #     # base_mesh = load_objs_as_meshes(['dataset/cvpr_final/maple_1122/base_shape/maple_5.obj'])
    #     save_dir = 'results/cvpr/rebuttal/fitting'
    #     os.makedirs(save_dir,exist_ok=True)
    #     save_name = os.path.join(save_dir, f'{filename}_fit.obj')
    # # reconstructor.neuraleaf_fitting(points)
    #     # if not os.path.exists(save_name):
    #     reconstructor.fitting_direct(base_mesh, points,0,save_name,'20',epoch=601)
    #         # reconstructor.neuraleaf_fitting(points,save_name,epoch=1000)
    