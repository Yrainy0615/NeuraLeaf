import torch
import numpy as np
from matplotlib import pyplot as plt
import trimesh
import cv2
import meshlib.mrmeshpy as mm
import meshlib.mrmeshnumpy as mn
import numpy as np
import math
from torchvision.utils import make_grid, save_image
from pytorch3d.structures import Meshes
from pytorch3d.io import IO
import pymeshlab
import sys
sys.path.append('scripts')
from data.mask_sdf import SDF2D
import os

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def latent_to_mask(latent, decoder, size,k=None):
    x_coord = np.linspace(0, 1, size).astype(np.float32)
    y_coord = np.linspace(0, 1, size).astype(np.float32)
    xy = np.stack(np.meshgrid(x_coord, y_coord), -1).reshape(-1, 2)
    xy = torch.tensor(xy, dtype=torch.float32).to(latent.device)
    xy = xy.unsqueeze(0).expand(latent.shape[0], -1, -1)

    glob_cond = torch.cat([latent.unsqueeze(1).expand(-1, xy.shape[1], -1), xy], dim=2)
    sdf = decoder(glob_cond)
    sdf_2d = sdf.view(latent.shape[0], size, size)
    # mask = sdf_to_mask(sdf_2d, k)
    mask = torch.nn.Hardsigmoid()(sdf_2d*k)
    return 1-mask


def sdf_to_mask(sdf:torch.Tensor,k:float=1):
    mask = 1/ (1+torch.exp(-k*sdf))
    return mask


def mask_to_mesh(masks:torch.tensor, shape_idx,save_mesh=False,save_folder ='results/cvpr/generation'):
    # list for store pytorch3d meshes
    os.makedirs(save_folder,exist_ok=True)
    if masks.dim() == 3 and masks.shape[0] == 1:
        masks = masks.unsqueeze(0)
    elif masks.dim() == 3 and masks.shape[0] > 1:
        masks = masks.unsqueeze(1)
    assert masks.dim() == 4, "masks should be 4D tensor"
    verts_list = []
    faces_list = []
    for i,mask in enumerate(masks):
        if mask.min() < 0:  
            mask = (mask+1)
        mask = mask[0]
        mask_numpy = mask.detach().cpu().numpy()
        y, x = np.nonzero(mask_numpy > 0)
        y_uv = y / mask_numpy.shape[0]
        x_uv = x / mask_numpy.shape[1]
        z = np.zeros_like(x)
        verts = np.stack([x_uv, y_uv, z],axis=-1).reshape(-1,3)
        # verts = verts - verts.mean(axis=0)
        pc = mn.pointCloudFromPoints(verts)
        # pc.validPoints = mm.pointUniformSampling(pc, 1e-2)
        pc.invalidateCaches()
        mesh = mm.triangulatePointCloud(pc)

        out_faces = mn.getNumpyFaces(mesh.topology)
        verts = mn.getNumpyVerts(mesh)
        # mesh_save = trimesh.Trimesh(verts, out_faces, process=False)
        verts_list.append(verts)
        faces_list.append(out_faces)
        if type(shape_idx) == int:
            shape_idx = [shape_idx]
        if save_mesh:
            current_mesh = Meshes(verts=[torch.tensor(verts).float()], faces=[torch.tensor(out_faces).long()])
            IO().save_mesh(current_mesh, os.path.join(save_folder,f'base_{shape_idx[i]}.obj'))
    mesh_pytorch3d = Meshes(verts=[torch.tensor(verts).float() for verts in verts_list], faces=[torch.tensor(faces).long() for faces in faces_list])

    return mesh_pytorch3d
        
def mask_to_mesh_distancemap(mask_file:str,mode='nparray'):
    # get folder dir of maskfile
    mask_dir = os.path.dirname(mask_file) 
    mask =cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)/255
    sdf_name = os.path.basename(mask_file).split(".")[0]
    sdf_path = os.path.join(mask_dir,'output' ,sdf_name+'_sdf.png')
    if mode == 'distance_map':
        if not os.path.exists(sdf_path):
            mySDF2D = SDF2D(mask_file)              
            sdf_img = mySDF2D.mask2sdf()
    
        distance_map = mm.loadDistanceMapFromImage(mm.Path(sdf_path), 0)
        polyline2 = mm.distanceMapTo2DIsoPolyline(distance_map, isoValue=12)

        # Create an empty HolesVertIds object if there are no holes
        holes_vert_ids = mm.HolesVertIds()

        # Call contours2 with the required argument
        contours = polyline2.contours2(holes_vert_ids)

        # Compute the triangulation inside the contour
        mesh = mm.triangulateContours(contours)
    elif mode == 'nparray':
        y_indices, x_indices = np.where(mask == 1)
        z_coords = np.zeros_like(x_indices)
        u_coords = x_indices / mask.shape[1]
        v_coords = (y_indices / mask.shape[0])
        verts = np.stack([u_coords, v_coords, z_coords], axis=-1)
        # random sample 15000 points
        idx = np.random.choice(verts.shape[0], 15000, replace=False)
        verts = verts[idx]
        # move mean to  origin
        center = verts.mean(axis=0)
        verts -= center
        pc =mn.pointCloudFromPoints(verts)
        
        pc.validPoints = mm.pointUniformSampling(pc, 1e-3)
        pc.invalidateCaches()
        mesh = mm.triangulatePointCloud(pc)
        # simplify mesh
        # mesh.meshing_decimation_clustering()
        # mesh = mm.offsetMesh(mesh, 0.0)
    # props = mm.SubdivideSettings()
    # props.maxDeviationAfterFlip = 0.5
    # mm.subdivideMesh(mesh,props)
    verts = mn.getNumpyVerts(mesh)
    faces = mn.getNumpyFaces(mesh.topology)
    mesh_torch3d = Meshes(verts=[torch.tensor(verts).float()], faces=[torch.tensor(faces).long()])
    return mesh_torch3d
    # sdf = image_to_sdf(sdf_img)
    

def save_tensor_image(tensor,path,normalize=True):
    grid  = make_grid(tensor.float(), n_row=int(math.sqrt(tensor.shape[0])), normalize=True)
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    save_image(grid, path,normalize=normalize)

def denormalize(tensor, mean=0.5,std=0.5):
    mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.tensor(std, dtype=tensor.dtype, device=tensor.device)
    return tensor * std + mean
    
    

def deform_mesh(mesh,
                deformer,
                lat_rep,
                anchors=None,
                lat_rep_shape=None):
    points_neutral = torch.from_numpy(np.array(mesh.vertices)).float().unsqueeze(0).to(lat_rep.device)

    with torch.no_grad():
        grid_points_split = torch.split(points_neutral, 5000, dim=1)
        delta_list = []
        for split_id, points in enumerate(grid_points_split):
            if lat_rep_shape is None:
                glob_cond = lat_rep.repeat(1, points.shape[1], 1)
            else:
                glob_cond = torch.cat([lat_rep_shape, lat_rep], dim=-1)
                glob_cond = glob_cond.repeat(1, points.shape[1], 1)
            if anchors is not None:
                d, _ = deformer(points, glob_cond, anchors.unsqueeze(1).repeat(1, points.shape[1], 1, 1))
            else:
                d = deformer(points, glob_cond)
            delta_list.append(d.detach().clone())

            torch.cuda.empty_cache()
        delta = torch.cat(delta_list, dim=1)

    pred_posed = points_neutral[:, :, :3] + delta.squeeze()
    verts = pred_posed.detach().cpu().squeeze().numpy()
    mesh_deformed = trimesh.Trimesh(verts, mesh.faces, process=False)

    return mesh_deformed

if __name__ == "__main__":
    mask_path = "dataset/LeafData/Chinar/healthy/Chinar_healthy_0001_mask_aligned.JPG"
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask_tensor = torch.tensor(mask).unsqueeze(0).unsqueeze(0)
    mesh = mask_to_mesh(mask_tensor) 
    mesh = mask_to_mesh_distancemap(mask_path)
    # mrmeshpy.saveMesh(mesh, mrmeshpy.Path("leaf_2d.ply"))
    
    
    