import torch
import argparse
import yaml
import os
from scripts.models.neuralbs import NBS
import torch.nn.functional as F
from pytorch3d.structures import Meshes
from pytorch3d.loss import chamfer_distance, mesh_edge_loss, mesh_laplacian_smoothing
from scripts.utils.utils import latent_to_mask
from pytorch3d import transforms
from pytorch3d.io import IO, load_objs_as_meshes, load_ply, load_obj
import trimesh
import numpy as np
import meshlib.mrmeshpy as mm
import meshlib.mrmeshnumpy as mn
from scripts.models.decoder import UDFNetwork, SWPredictor, TransPredictor
from scripts.models.encoder import ShapeEncoder
from scripts.utils.geom_utils import  points_to_voxel, raw_to_canonical, mask2mesh
from scripts.utils.loss_utils import ARAPLoss
from probreg import cpd


def rigid_cpd_cuda(basepoints: torch.tensor, deformed_points: torch.tensor, use_cuda=True):
    """Rigid CPD registration using CUDA if available."""
    import cupy as cp
    if use_cuda:
        to_cpu = cp.asnumpy
        cp.cuda.Device().use()
        cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
    else:
        cp = np
        to_cpu = lambda x: x
    source_pt = cp.asarray(basepoints)
    target_pt = cp.asarray(deformed_points)
    if source_pt.shape[0] > 10000:
        random_index = np.random.choice(source_pt.shape[0], 10000, replace=False)
        source_pt = source_pt[random_index]
    rcpd = cpd.RigidCPD(target_pt, use_cuda=use_cuda)
    tf_param_rgd, _, _ = rcpd.registration(source_pt)
    target_rgd = tf_param_rgd.transform(target_pt)
    return target_rgd


class Reconstructor:
    def __init__(self, cfg_deform, device, shape_decoder, sw_predictor, trans_predictor, 
                 shape_codes, deform_codes, bone, k, shape_encoder=None, deform_encoder=None, use_length_reg=False, use_arap=False):
        self.device = device
        self.cfg_deform = cfg_deform['Trainer']
        self.nbs_model = NBS(cfg_deform['NBS'])
        self.shape_decoder = shape_decoder
        self.sw_predictor = sw_predictor
        self.trans_predictor = trans_predictor
        self.shape_codes = shape_codes.to(self.device) if shape_codes is not None else None
        self.deform_codes = deform_codes.to(self.device) if deform_codes is not None else None
        self.bone = bone
        self.k = k.to(self.device) if isinstance(k, torch.Tensor) else k
        self.shape_encoder = shape_encoder
        self.deform_encoder = deform_encoder
        self.use_length_reg = use_length_reg
        self.use_arap = use_arap
        
        # Compute mean codes if encoders are not provided
        if self.shape_encoder is None and self.shape_codes is not None:
            self.mean_shape_code = self.shape_codes.mean(dim=0)
        else:
            self.mean_shape_code = None
            
        if self.deform_encoder is None and self.deform_codes is not None:
            self.mean_deform_code = self.deform_codes.mean(dim=0)
        else:
            self.mean_deform_code = None

    def normalize_mesh(self, points):
        """Normalize mesh points to [0, 1] range."""
        # Centering
        points = points - points.mean(0)
        # Scaling
        scale = torch.max(torch.abs(points))
        if scale > 0:
            points = points / scale
            # Shift to [0, 1] range
            points = (points + 1.0) / 2.0
        else:
            # If all points are the same, center them at 0.5
            points = points + 0.5
        return points

    def get_initial_codes(self, points=None):
        """Get initial shape and deform codes from encoder or use mean."""
        if  self.shape_encoder is not None and self.deform_encoder is not None:
            voxel = points_to_voxel(points)
            shape_code = self.shape_encoder(voxel).squeeze(0)  # [code_dim]
            deform_code = self.deform_encoder(voxel).squeeze(0)  # [code_dim]
        else:
            shape_code = self.shape_codes.mean(dim=0)
            deform_code = self.deform_codes.mean(dim=0)
            
        return shape_code, deform_code

    def rigid_to_canonical(self, points, base_points, save_folder=None,save_name=None):
        """Rigid registration of points to canonical base shape."""
        # check if points saved then load 
        rigid_save_path = os.path.join(save_folder, f'{save_name}_ori.obj')
        if os.path.exists(rigid_save_path):
            registered_points = load_objs_as_meshes([rigid_save_path])
            return registered_points.verts_list()[0]
        else:
            points_np = points.detach().cpu().numpy()
            base_points_np = base_points.detach().cpu().numpy()
            registered_points = rigid_cpd_cuda(base_points_np, points_np, use_cuda=True)
            return torch.tensor(registered_points, dtype=torch.float32).to(self.device)


    def single_leaf_predict(self, mesh_path):
        """
        Single leaf prediction pipeline:
        1. Read mesh
        2. Normalize to [0, 1]
        3. Get base shape from shape decoder (mask)
        4. Convert mask to mesh
        5. Rigid registration to canonical base shape
        """
        # 1. Read mesh
        if mesh_path.endswith('.obj'):
            mesh = load_objs_as_meshes([mesh_path])
            verts = mesh.verts_packed()
        elif mesh_path.endswith('.ply'):
            mesh_data = load_ply(mesh_path)
            verts = mesh_data[0]
        
        # 2. Normalize to [0, 1]
        verts_normalized = self.normalize_mesh(verts)
        
        # 3. Get initial codes
        shape_code, deform_code = self.get_initial_codes(verts_normalized)
        
        # 4. Get base shape from shape decoder (mask)
        base_mask = latent_to_mask(shape_code.unsqueeze(0), self.shape_decoder, size=256, k=self.k)
        
        # 5. Convert mask to mesh
        base_mesh = mask2mesh([base_mask])
        
        # 6. Rigid registration to canonical base shape
        base_points = base_mesh.verts_packed()
        points_registered = self.rigid_to_canonical(verts_normalized, base_points, save_folder=args.save_folder, save_name=mesh_path.split('/')[-1].split('.')[0])
        
        return {
            'base_mesh': base_mesh,
            'registered_points': points_registered,
            'shape_code': shape_code,
            'deform_code': deform_code
        }

    def deform_lbs(self, bone, base_points, sw, rts_fw):
        """
        Linear Blend Skinning deformation.
        rts_fw: (b, K, 7) - rotation, translation, scale
        sw: (b, N, K) - skinning weights
        bone: (K, 3) - bone positions
        """
        bone = bone.unsqueeze(0).expand(base_points.shape[0], -1, -1).to(self.device)
        B, N, K = sw.shape
        v0 = base_points.view(-1, 3)
        disp = rts_fw[:, :, :3]
        rot = rts_fw[:, :, 3:]
        rot = transforms.quaternion_to_matrix(rot.view(B * K, 4).contiguous()).view(B, K, 3, 3).contiguous()
        hd_disp = torch.repeat_interleave(disp, N, dim=0)
        hd_rot = torch.repeat_interleave(rot, N, dim=0)
        hd_pos = torch.repeat_interleave(bone, N, dim=0)
        per_hd_v = torch.einsum("abcd,abd->abc", hd_rot, (v0[:, None] - hd_pos)) + hd_pos + hd_disp
        region_score = sw.view(-1, K)
        v = torch.sum(region_score[:, :, None] * per_hd_v, 1)
        return v.view(B, N, 3)

    def reduce_lr(self, optimizer, factor=0.9):
        """Reduce learning rate."""
        for param_group in optimizer.param_groups:
            param_group['lr'] *= factor

    def find_boundary_edges(self, faces):
        """Find boundary edges of the mesh."""
        edge_count = {}
        for face in faces:
            edges = [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]
            for edge in edges:
                edge = tuple(sorted(edge))
                if edge not in edge_count:
                    edge_count[edge] = 0
                edge_count[edge] += 1
        # Find edges that belong to only one face (boundary edges)
        boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
        return boundary_edges

    def compute_edge_length(self, verts, edges):
        """Compute edge lengths for given edges."""
        lengths = []
        for start_idx, end_idx in edges:
            start_point = verts[start_idx]
            end_point = verts[end_idx]
            edge_length = torch.norm(start_point - end_point)
            lengths.append(edge_length)
        return torch.stack(lengths) if lengths else torch.tensor([], device=verts.device)

    def neuraleaf_fitting(self, base_mesh, points, epoch=300, save_mesh=False, save_path=None, 
                         shape_code_init=None, deform_code_init=None, input_mesh=None):
        """
        Neural leaf fitting using shape and deform codes.
        
        Args:
            base_mesh: Base mesh for fitting
            points: Target points to fit
            epoch: Number of optimization epochs
            save_mesh: Whether to save the fitted mesh
            save_path: Path to save the mesh
            shape_code_init: Initial shape code (optional)
            deform_code_init: Initial deform code (optional)
            input_mesh: Input mesh to initialize codes from (optional)
        """
        # Get initial codes
        if shape_code_init is None or deform_code_init is None:
            if input_mesh is not None:
                shape_code_init, deform_code_init = self.get_initial_codes(input_mesh)
            else:
                shape_code_init, deform_code_init = self.get_initial_codes(base_mesh)
        
        # Initialize as parameters
        shape_code = torch.nn.Parameter(shape_code_init.clone())
        deform_code = torch.nn.Parameter(deform_code_init.clone())
        
        optimizer_shape = torch.optim.Adam([shape_code], lr=0.01) 
        optimizer_deform = torch.optim.Adam([deform_code], lr=0.005) 
        
        deform_points = points.to(self.device)
        if deform_points.ndimension() == 2:
            deform_points = deform_points.unsqueeze(0)
        # Normalize deform_points to zero mean
        deform_points = raw_to_canonical(deform_points.squeeze(0)).unsqueeze(0)
        
        base_points = base_mesh.verts_packed().unsqueeze(0).to(self.device)
        # Normalize base_points to zero mean
        base_points = raw_to_canonical(base_points.squeeze(0)).unsqueeze(0)
        base_faces = base_mesh.faces_packed().unsqueeze(0).to(self.device)
        arap_loss = ARAPLoss(base_points, base_faces)
        if self.use_length_reg:
            boundary_edges = self.find_boundary_edges(base_faces.squeeze(0))
            initial_edge_lengths = self.compute_edge_length(base_points.squeeze(0), boundary_edges)
        else:
            initial_edge_lengths = None
        
        for i in range(epoch):          
            optimizer_shape.zero_grad()
            optimizer_deform.zero_grad()
            
            # Update base shape from shape_code
            with torch.no_grad():
                base_mask = latent_to_mask(shape_code.unsqueeze(0), self.shape_decoder, size=256, k=self.k)
                base_mesh_new = mask2mesh([base_mask])
                base_mesh_new = base_mesh_new.to(self.device)
                base_points = base_mesh_new.verts_packed().unsqueeze(0)
                # Normalize base_points to zero mean
                base_points = raw_to_canonical(base_points.squeeze(0)).unsqueeze(0)
                base_faces = base_mesh_new.faces_packed().unsqueeze(0)
            
            
            # Forward pass
            rts_fw = self.trans_predictor(deform_code.unsqueeze(0), self.bone)
            sw = self.sw_predictor(shape_code.unsqueeze(0), base_points)
            v0 = self.deform_lbs(self.bone, base_points, sw, rts_fw)
            
            new_mesh = Meshes(verts=v0, faces=base_faces, textures=base_mesh.textures)
            loss_chamfer = chamfer_distance(deform_points, v0)[0]
            
            # Shape regularizers: ARAP, Laplacian, and edge length
            # ARAP loss: As-Rigid-As-Possible regularization
            if self.use_arap:
                loss_arap = arap_loss(v0, base_points)[0] 
            else:
                loss_arap = torch.tensor(0.0, device=self.device)
            if self.use_length_reg:
                new_edge_lengths = self.compute_edge_length(v0.squeeze(0), boundary_edges)
                loss_length = (new_edge_lengths - initial_edge_lengths).abs().mean()
            else:
                loss_length = torch.tensor(0.0, device=self.device)
            
            # Laplacian loss: mesh smoothness
            loss_laplacian = mesh_laplacian_smoothing(new_mesh)
            loss_edge = mesh_edge_loss(new_mesh)
            
            # Edge length loss: preserve boundary edge lengths
            loss_length = torch.tensor(0.0, device=self.device)
            
            loss_total = loss_chamfer + loss_arap + loss_laplacian + loss_edge + loss_length
            loss_total.backward()
            optimizer_shape.step()
            optimizer_deform.step()
            if i % 50 == 0:
                print(f'Epoch {i} loss: {loss_total.item():.6f}, chamfer: {loss_chamfer.item():.6f}, smooth: {loss_laplacian.item():.6f}, edge: {loss_edge.item():.6f}, length: {loss_length.item():.6f},arap: {loss_arap.item():.6f}')
            if i % (epoch//3) == 0 and epoch !=0:
                self.reduce_lr(optimizer_shape)
                self.reduce_lr(optimizer_deform)
    
        if save_mesh and save_path is not None:
            IO().save_mesh(new_mesh, save_path)
            # Save final base mesh from updated shape_code
            with torch.no_grad():
                final_base_mask = latent_to_mask(shape_code.unsqueeze(0), self.shape_decoder, size=256, k=self.k)
                final_base_mesh = mask2mesh([final_base_mask])
                IO().save_mesh(final_base_mesh, save_path.replace('.obj', '_base.obj'))
            deform_verts = deform_points.squeeze(0)  # [N, 3]
            deform_faces = torch.zeros((0, 3), dtype=torch.long, device=deform_verts.device)  # [0, 3]
            deform_points_mesh = Meshes(verts=[deform_verts], faces=[deform_faces])
            IO().save_mesh(deform_points_mesh, save_path.replace('_fitted.obj', '_ori.obj'))
        return loss_chamfer, new_mesh

    def fitting_direct(self, base_mesh, deform_points, epoch=601, save_mesh=False, save_path=None):
        """
        Direct fitting by optimizing skinning weights, bones, and transformations.
        """
        base_points = base_mesh.verts_packed().unsqueeze(0)
        base_faces = base_mesh.faces_packed().unsqueeze(0)
        # Normalize base_points to zero mean
        base_points = raw_to_canonical(base_points.squeeze(0)).unsqueeze(0)
        base_points = base_points * 2  # Rescale base points
        shape_code, _ = self.get_initial_codes(base_points)
        deform_points = deform_points.to(self.device)
        if deform_points.ndimension() == 2:
            deform_points = deform_points.unsqueeze(0)
        # Normalize deform_points to zero mean
        deform_points = raw_to_canonical(deform_points.squeeze(0)).unsqueeze(0)
        
        b, N, _ = base_points.size()
        
        # Initialize bone if not provided
        if self.bone is None:
            bone_tensor = self.nbs_model.generate_bone()
            bone_tensor = bone_tensor[:, :3]
            bone = torch.nn.Parameter(bone_tensor.to(self.device))
        else:
            bone = torch.nn.Parameter(self.bone.clone())
        arap_loss = ARAPLoss(base_points, base_faces)
        K = bone.size(0)
        sw = torch.nn.Parameter(torch.rand(N, K).uniform_(0.1, 1.0).to(self.device))
        t = torch.zeros(K, 7).to(self.device)
        t[:, 3] = 1  # Quaternion w component
        T = torch.nn.Parameter(t)
        
        optimizer_sw = torch.optim.Adam([sw], lr=0.1)
        optimizer_bone = torch.optim.Adam([bone], lr=0.5)
        optimizer_T = torch.optim.Adam([T], lr=0.1)

        # Calculate the epoch threshold for shape-related parameter updates
        
        for i in range(epoch):
            optimizer_bone.zero_grad()
            optimizer_sw.zero_grad()
            optimizer_T.zero_grad()
            
            sw_softmax = F.softmax(sw, dim=1).unsqueeze(0)
            T = T.to(self.device)
            base_points = base_points.to(self.device)
            base_faces = base_faces.to(self.device)
            
            # Forward pass
            v0 = self.deform_lbs(bone, base_points, sw_softmax, T.unsqueeze(0))
            new_mesh = Meshes(verts=v0, faces=base_faces, textures=base_mesh.textures)
            
            loss_chamfer = chamfer_distance(deform_points, v0)[0]
            loss_edge = mesh_edge_loss(new_mesh)
            loss_laplacian = mesh_laplacian_smoothing(new_mesh)
            
            loss_dict = {'shape': loss_chamfer, 'edge': loss_edge, 'smooth': loss_laplacian}
            loss_total = 0
            for key in loss_dict.keys():
                loss_total += loss_dict[key] * self.cfg_deform['lambdas'][key]
            
            loss_total.backward()
            optimizer_sw.step()
            optimizer_bone.step()
            optimizer_T.step()

            
            if i % 50 == 0:
                print(f'Epoch {i} loss: {loss_total.item():.6f}, chamfer: {loss_chamfer.item():.6f}, '
                      f'edge: {loss_edge.item():.6f}, smooth: {loss_laplacian.item():.6f}')
            
            if i % 300 == 0:
                self.reduce_lr(optimizer_sw)
                self.reduce_lr(optimizer_bone)
                self.reduce_lr(optimizer_T)
        
        if save_mesh and save_path is not None:
            IO().save_mesh(new_mesh, save_path)
        
        return loss_chamfer, new_mesh


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural Leaf Fitting')
    parser.add_argument('--gpu', type=int, default=0, help='gpu index')
    parser.add_argument('--save_folder', type=str, default='results/fitting', help='output directory')
    parser.add_argument('--config', type=str, default='scripts/configs/bashshape.yaml', help='config file')
    parser.add_argument('--config_deform', type=str, default='scripts/configs/deform.yaml', help='deform config file')
    parser.add_argument('--mesh_path', type=str, required=False, default='/mnt/data/cvpr_final/deform_train/17_deform.ply', help='path to input mesh')
    parser.add_argument('--method', type=str, default='neuraleaf', choices=['neuraleaf', 'direct'], 
                        help='fitting method')
    parser.add_argument('--epoch', type=int, default=1000, help='number of epochs')
    parser.add_argument('--use_length_reg', action='store_true', help='use length regularization')
    parser.add_argument('--use_arap', action='store_true', help='use arap loss')
    parser.add_argument('--shape_checkpoint', type=str, default='checkpoints/baseshape.pth', help='path to shape checkpoint')
    parser.add_argument('--deform_checkpoint', type=str, default='checkpoints/deform.pth', help='path to deform checkpoint')
    parser.add_argument('--shape_encoder_checkpoint', type=str, default='checkpoints/shape_encoder.pth', help='path to shape encoder checkpoint')
    parser.add_argument('--deform_encoder_checkpoint', type=str, default='checkpoints/deform_encoder.pth', help='path to deform encoder checkpoint')
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    CFG = yaml.safe_load(open(args.config, 'r'))
    CFG_deform = yaml.safe_load(open(args.config_deform, 'r'))
    
    # Load shape model
    shape_decoder = UDFNetwork(
        d_in=CFG['Base']['Z_DIM'],
        d_hidden=CFG['Base']['decoder_hidden_dim'],
        d_out=CFG['Base']['decoder_out_dim'],
        n_layers=CFG['Base']['decoder_nlayers'],
        udf_type='sdf',
        geometric_init=False
    )
    shape_decoder.eval()
    shape_decoder.to(device)
    
    checkpoint_base = torch.load(args.shape_checkpoint, map_location='cpu') #231
    shape_decoder.load_state_dict(checkpoint_base['decoder'])
    k = checkpoint_base['k']
    shape_codes = checkpoint_base['latent_shape']['weight']
    
    # Load deformation model
    num_bones = CFG_deform['NBS']['num_bones']
    deform_latent_dim = CFG_deform['NBS']['lat_dim']  # Deform code dimension
    shape_latent_dim = CFG['Base']['Z_DIM']  # Shape code dimension
    swpredictor = SWPredictor(num_bones=num_bones, latent_dim=shape_latent_dim).to(device)
    transpredictor = TransPredictor(num_bones=num_bones, latent_dim=128).to(device)
    checkpoint_deform = torch.load(args.deform_checkpoint, map_location='cpu') # 152 latest_deform_shape_prior
    bone = checkpoint_deform['bone'].to(device)
    deform_codes = checkpoint_deform['deform_codes']['weight']
    
    swpredictor.load_state_dict(checkpoint_deform['SWpredictor'])
    transpredictor.load_state_dict(checkpoint_deform['Transpredictor'])
    swpredictor.eval()
    transpredictor.eval()
    
    # Initialize encoders (optional)
    shape_encoder = None  # Set to ShapeEncoder instance if available
    deform_encoder = None  # Set to DeformEncoder instance if available
    if os.path.exists(args.shape_encoder_checkpoint) and os.path.exists(args.deform_encoder_checkpoint):
        shape_encoder = ShapeEncoder(code_dim=shape_latent_dim, res=128, output_dim=shape_latent_dim).to(device)
        deform_encoder = ShapeEncoder(code_dim=deform_latent_dim, res=128, output_dim=deform_latent_dim).to(device)
        shape_encoder.load_state_dict(torch.load(args.shape_encoder_checkpoint, map_location='cpu'))
        deform_encoder.load_state_dict(torch.load(args.deform_encoder_checkpoint, map_location='cpu'))
        shape_encoder.eval()
        deform_encoder.eval()

    # Create reconstructor
    reconstructor = Reconstructor(
        CFG_deform, device, shape_decoder, swpredictor, transpredictor,
        shape_codes, deform_codes, bone, k,
        shape_encoder=shape_encoder, deform_encoder=deform_encoder,
        use_length_reg=args.use_length_reg,
        use_arap=args.use_arap
    )
    
    if args.mesh_path is not None:
        # Extract filename from input mesh path (without extension)
        mesh_basename = os.path.splitext(os.path.basename(args.mesh_path))[0]
        
        # Create output directory if it doesn't exist
        os.makedirs(args.save_folder, exist_ok=True)
        
        # Single leaf prediction
        result = reconstructor.single_leaf_predict(args.mesh_path)
        
        # Fitting
        if args.method == 'neuraleaf':
            save_path = os.path.join(args.save_folder, f'{mesh_basename}_fitted.obj')
            loss, mesh = reconstructor.neuraleaf_fitting(
                result['base_mesh'], result['registered_points'],
                epoch=args.epoch, save_mesh=True,
                save_path=save_path,
                shape_code_init=result['shape_code'],
                deform_code_init=result['deform_code']
            )
        else:
            save_path = os.path.join(args.save_folder, f'{mesh_basename}_fitted_direct.obj')
            loss, mesh = reconstructor.fitting_direct(
                result['base_mesh'], result['registered_points'],
                epoch=args.epoch, save_mesh=True,
                save_path=save_path
            )
        print(f'Fitting completed. Chamfer distance: {loss.item():.6f}')
        print(f'Output files saved to: {args.save_folder}')

