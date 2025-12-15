import os
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
sys.path.insert(0, project_root)

import torch
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d import transforms
import meshlib.mrmeshpy as mm
import meshlib.mrmeshnumpy as mn
from scripts.utils.utils import latent_to_mask
from scripts.models.decoder import SWPredictor, TransPredictor
from scripts.utils.geom_utils import mask2mesh


class Generator:
    def __init__(self, device, shape_decoder, sw_predictor, trans_predictor, 
                 bone, shape_codes, deform_codes, k):
        """
        Initialize Generator.
        
        Args:
            device: torch device
            shape_decoder: shape decoder model
            sw_predictor: skinning weights predictor
            trans_predictor: transformation predictor
            bone: bone positions (K, 3)
            shape_codes: shape codes (N, code_dim)
            deform_codes: deformation codes (M, code_dim)
            k: temperature parameter for mask generation
        """
        self.device = device
        self.shape_decoder = shape_decoder
        self.sw_predictor = sw_predictor
        self.trans_predictor = trans_predictor
        self.bone = bone.to(device) if isinstance(bone, torch.Tensor) else bone
        self.shape_codes = shape_codes.to(device) if isinstance(shape_codes, torch.Tensor) else shape_codes
        self.deform_codes = deform_codes.to(device) if isinstance(deform_codes, torch.Tensor) else deform_codes
        self.k = k
        
    def deform_lbs(self, base_points, sw, rts_fw):
        """
        Linear Blend Skinning deformation.
        
        Args:
            base_points: (B, N, 3) base mesh points
            sw: (B, N, K) skinning weights
            rts_fw: (B, K, 7) transformations [translation(3), rotation(4)]
        
        Returns:
            deformed_points: (B, N, 3) deformed points
        """
        bone = self.bone.unsqueeze(0).expand(base_points.shape[0], -1, -1).to(self.device)
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
    
    def random_generation(self, shape_idx=None, deform_idx=None):
        """
        Random generation: generate a single mesh from random or given shape and deform codes.
        
        Args:
            shape_idx: optional shape code index, if None randomly select
            deform_idx: optional deform code index, if None randomly select
        
        Returns:
            mesh: PyTorch3D Meshes object (single mesh)
        """
        # Select indices
        if shape_idx is None:
            shape_idx = np.random.randint(0, self.shape_codes.shape[0])
        if deform_idx is None:
            deform_idx = np.random.randint(0, self.deform_codes.shape[0])
        
        # Get codes
        shape_code = self.shape_codes[shape_idx].unsqueeze(0).to(self.device)
        deform_code = self.deform_codes[deform_idx].unsqueeze(0).to(self.device)
        
        # Generate base mesh from shape code
        with torch.no_grad():
            base_mask = latent_to_mask(shape_code, self.shape_decoder, k=self.k, size=256)
            base_mesh = mask2mesh([base_mask])
            base_mesh = base_mesh.to(self.device)
            
            # Get base points and center them
            base_points = base_mesh.verts_packed().unsqueeze(0)
            base_points = base_points - base_points.mean(1, keepdim=True)
            base_faces = base_mesh.faces_packed().unsqueeze(0)
            
            # Apply deformation
            sw = self.sw_predictor(latent_code=shape_code, centers=base_points)
            rts_fw = self.trans_predictor(latent_code=deform_code, centers=self.bone)
            deformed_points = self.deform_lbs(base_points, sw, rts_fw)
            
            deformed_mesh = Meshes(deformed_points, base_faces)
        
        return deformed_mesh
    
    def shape_interpolation(self, shape_idx1=None, shape_idx2=None, deform_idx=None, n_steps=5):
        """
        Shape interpolation: interpolate between two shape codes with fixed deform code.
        
        Args:
            shape_idx1: optional first shape code index, if None randomly select
            shape_idx2: optional second shape code index, if None randomly select
            deform_idx: optional deform code index, if None randomly select
            n_steps: number of interpolation steps (default 5)
        
        Returns:
            meshes: list of PyTorch3D Meshes objects (n_steps meshes)
        """
        # Select indices
        if shape_idx1 is None:
            shape_idx1 = np.random.randint(0, self.shape_codes.shape[0])
        if shape_idx2 is None:
            shape_idx2 = np.random.randint(0, self.shape_codes.shape[0])
            # Ensure different indices
            while shape_idx2 == shape_idx1:
                shape_idx2 = np.random.randint(0, self.shape_codes.shape[0])
        if deform_idx is None:
            deform_idx = np.random.randint(0, self.deform_codes.shape[0])
        
        # Get codes
        shape_code1 = self.shape_codes[shape_idx1].to(self.device)
        shape_code2 = self.shape_codes[shape_idx2].to(self.device)
        deform_code = self.deform_codes[deform_idx].unsqueeze(0).to(self.device)
        
        # Interpolation weights
        interp_weights = torch.linspace(0, 1, n_steps).to(self.device)
        
        meshes = []
        with torch.no_grad():
            for i, alpha in enumerate(interp_weights):
                # Interpolate shape codes
                shape_code = shape_code1 * (1 - alpha) + shape_code2 * alpha
                shape_code = shape_code.unsqueeze(0)
                
                # Generate base mesh
                base_mask = latent_to_mask(shape_code, self.shape_decoder, k=self.k, size=256)
                base_mesh = mask2mesh(base_mask)
                base_mesh = base_mesh.to(self.device)
                
                # Get base points and center them
                base_points = base_mesh.verts_packed().unsqueeze(0)
                base_points = base_points - base_points.mean(1, keepdim=True)
                base_faces = base_mesh.faces_packed().unsqueeze(0)
                
                # Apply deformation
                sw = self.sw_predictor(latent_code=shape_code, centers=base_points)
                rts_fw = self.trans_predictor(latent_code=deform_code, centers=self.bone)
                deformed_points = self.deform_lbs(base_points, sw, rts_fw)
                
                deformed_mesh = Meshes(deformed_points, base_faces)
                meshes.append(deformed_mesh)
        
        return meshes
    
    def deformation_interpolation(self, shape_idx=None, deform_idx1=None, deform_idx2=None, n_steps=5):
        """
        Deformation interpolation: interpolate between two deform codes with fixed shape code.
        
        Args:
            shape_idx: optional shape code index, if None randomly select
            deform_idx1: optional first deform code index, if None randomly select
            deform_idx2: optional second deform code index, if None randomly select
            n_steps: number of interpolation steps (default 5)
        
        Returns:
            meshes: list of PyTorch3D Meshes objects (n_steps meshes)
        """
        # Select indices
        if shape_idx is None:
            shape_idx = np.random.randint(0, self.shape_codes.shape[0])
        if deform_idx1 is None:
            deform_idx1 = np.random.randint(0, self.deform_codes.shape[0])
        if deform_idx2 is None:
            deform_idx2 = np.random.randint(0, self.deform_codes.shape[0])
            # Ensure different indices
            while deform_idx2 == deform_idx1:
                deform_idx2 = np.random.randint(0, self.deform_codes.shape[0])
        
        # Get codes
        shape_code = self.shape_codes[shape_idx].unsqueeze(0).to(self.device)
        deform_code1 = self.deform_codes[deform_idx1].to(self.device)
        deform_code2 = self.deform_codes[deform_idx2].to(self.device)
        
        # Interpolation weights
        interp_weights = torch.linspace(0, 1, n_steps).to(self.device)
        
        # Generate base mesh once (fixed shape)
        with torch.no_grad():
            base_mask = latent_to_mask(shape_code, self.shape_decoder, k=self.k, size=256)
            base_mesh = mask2mesh([base_mask])
            base_mesh = base_mesh.to(self.device)
            
            # Get base points and center them
            base_points = base_mesh.verts_packed().unsqueeze(0)
            base_points = base_points - base_points.mean(1, keepdim=True)
            base_faces = base_mesh.faces_packed().unsqueeze(0)
            
            # Get skinning weights (fixed for shape)
            sw = self.sw_predictor(latent_code=shape_code, centers=base_points)
        
        meshes = []
        with torch.no_grad():
            for i, alpha in enumerate(interp_weights):
                # Interpolate deform codes
                deform_code = deform_code1 * (1 - alpha) + deform_code2 * alpha
                deform_code = deform_code.unsqueeze(0)
                
                # Apply deformation
                rts_fw = self.trans_predictor(latent_code=deform_code, centers=self.bone)
                deformed_points = self.deform_lbs(base_points, sw, rts_fw)
                
                deformed_mesh = Meshes(deformed_points, base_faces)
                meshes.append(deformed_mesh)
        
        return meshes


if __name__ == '__main__':
    import argparse
    import yaml
    from scripts.models.decoder import UDFNetwork
    from pytorch3d.io import IO
    
    parser = argparse.ArgumentParser(description='Generate encoder training dataset')
    parser.add_argument('--gpu', type=int, default=0, help='gpu index')
    parser.add_argument('--config', type=str, default='scripts/configs/bashshape.yaml', help='shape config file')
    parser.add_argument('--config_deform', type=str, default='scripts/configs/deform.yaml', help='deform config file')
    parser.add_argument('--shape_checkpoint', type=str, default='checkpoints/baseshape.pth', help='shape checkpoint path')
    parser.add_argument('--deform_checkpoint', type=str, default='checkpoints/deform.pth', help='deform checkpoint path')
    parser.add_argument('--output_dir', type=str, default='/mnt/data/encoder_dataset', help='output directory')
    parser.add_argument('--num_deforms_per_shape', type=int, default=5, help='number of random deform codes per shape')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--mode', type=str, default='random', choices=['random', 'shape_interp', 'deformation_interp'], help='generation mode')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    # Load configs
    CFG = yaml.safe_load(open(args.config, 'r'))
    CFG_deform = yaml.safe_load(open(args.config_deform, 'r'))
    
    print("Loading models...")
    
    # Load shape decoder
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
    
    checkpoint_base = torch.load(args.shape_checkpoint, map_location='cpu')
    shape_decoder.load_state_dict(checkpoint_base['decoder'])
    k = checkpoint_base.get('k', torch.tensor(1.0))
    shape_codes = checkpoint_base['latent_shape']['weight']
    
    # Load deformation model
    num_bones = CFG_deform['NBS']['num_bones']
    deform_latent_dim = CFG_deform['NBS']['lat_dim']  # Deform code dimension
    shape_latent_dim = CFG['Base']['Z_DIM']  # Shape code dimension
    sw_predictor = SWPredictor(num_bones=num_bones, latent_dim=shape_latent_dim).to(device)
    trans_predictor = TransPredictor(num_bones=num_bones, latent_dim=deform_latent_dim).to(device)
    
    checkpoint_deform = torch.load(args.deform_checkpoint, map_location='cpu')
    bone = checkpoint_deform['bone'].to(device)
    deform_codes = checkpoint_deform['deform_codes']['weight']
    
    sw_predictor.load_state_dict(checkpoint_deform['SWpredictor'])
    trans_predictor.load_state_dict(checkpoint_deform['Transpredictor'])
    sw_predictor.eval()
    trans_predictor.eval()
    
    print(f"Loaded {len(shape_codes)} shape codes and {len(deform_codes)} deform codes")
    
    # Create generator
    generator = Generator(
        device=device,
        shape_decoder=shape_decoder,
        sw_predictor=sw_predictor,
        trans_predictor=trans_predictor,
        bone=bone,
        shape_codes=shape_codes,
        deform_codes=deform_codes,
        k=k
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # random generation 
    if args.mode == 'random':
        num_shapes = len(shape_codes)
        num_deforms = len(deform_codes)
        total_meshes = num_shapes * args.num_deforms_per_shape
        
        print(f"Generating {total_meshes} meshes ({num_shapes} shapes × {args.num_deforms_per_shape} deforms each)...")
        
        mesh_io = IO()
        generated_count = 0
        
        for shape_idx in range(num_shapes):
            # Randomly select deform indices for this shape
            deform_indices = np.random.choice(num_deforms, args.num_deforms_per_shape, replace=False)
            
            for deform_idx in deform_indices:

                mesh = generator.random_generation(shape_idx=shape_idx, deform_idx=int(deform_idx))
                
                # Save mesh
                filename = f'{shape_idx}_{int(deform_idx)}.obj'
                save_path = os.path.join(args.output_dir, filename)
                mesh_io.save_mesh(mesh, save_path)
                
                generated_count += 1
                if generated_count % 10 == 0:
                    print(f"Progress: {generated_count}/{total_meshes} meshes generated")
    
    # shape interpolation
    if args.mode == 'shape_interp':
        pass

    # deformation interpolation
    if args.mode == 'deformation_interp':
        pass

    print(f"Dataset generation completed!")
    print(f"Total meshes generated: {generated_count}/{total_meshes}")
    print(f"Output directory: {args.output_dir}")

