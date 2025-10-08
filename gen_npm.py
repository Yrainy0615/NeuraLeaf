import os
import argparse
import yaml 
import torch
from scripts.models.decoder import UDFNetwork
import numpy as np
import trimesh
import skimage

def get_rotation_matrix(x_degrees=0, y_degrees=0, z_degrees=0):
    """
    生成绕 X, Y, Z 轴旋转的旋转矩阵。
    """
    x = np.deg2rad(x_degrees)
    y = np.deg2rad(y_degrees)
    z = np.deg2rad(z_degrees)
    
    Rx = np.array([
        [1, 0,           0,          0],
        [0, np.cos(x), -np.sin(x), 0],
        [0, np.sin(x),  np.cos(x), 0],
        [0, 0,           0,          1]
    ])
    
    Ry = np.array([
        [ np.cos(y), 0, np.sin(y), 0],
        [0,          1, 0,          0],
        [-np.sin(y), 0, np.cos(y), 0],
        [0,          0, 0,          1]
    ])
    
    Rz = np.array([
        [np.cos(z), -np.sin(z), 0, 0],
        [np.sin(z),  np.cos(z), 0, 0],
        [0,           0,          1, 0],
        [0,           0,          0, 1]
    ])
    
    rotation_matrix = Rz @ Ry @ Rx
    return rotation_matrix


def get_logits(decoder,
               encoding,
               grid_points,
               nbatch_points=100000,
               return_anchors=False):
    sample_points = grid_points.clone()

    grid_points_split = torch.split(sample_points, nbatch_points, dim=1)
    logits_list = []
    for points in grid_points_split:
        with torch.no_grad():
            inputs = torch.cat([points, encoding.repeat(1, points.shape[1], 1)], dim=-1)
            logits = decoder(inputs )
            logits = logits.squeeze()
            logits_list.append(logits.squeeze(0).detach().cpu())

    logits = torch.cat(logits_list, dim=0).numpy()
    if return_anchors:
        return logits
    else:
        return logits

def mesh_from_logits(logits, mini, maxi, resolution):
    logits = np.reshape(logits, (resolution,) * 3)

    #logits *= -1

    # padding to ba able to retrieve object close to bounding box bondary
    # logits = np.pad(logits, ((1, 1), (1, 1), (1, 1)), 'constant', constant_values=1000)
    threshold = 0.015
    # vertices, triangles = mcubes.marching_cubes(logits, threshold
    try:
        vertices, triangles,_,_ = skimage.measure.marching_cubes(logits, threshold)
    except:
        return None
        
    # rescale to original scale
    step = (np.array(maxi) - np.array(mini)) / (resolution - 1)
    vertices = vertices * np.expand_dims(step, axis=0)
    vertices += [mini[0], mini[1], mini[2]]

    return trimesh.Trimesh(vertices, triangles)

def create_grid_points_from_bounds(minimun, maximum, res, scale=None):
    if scale is not None:
        res = int(scale * res)
        minimun = scale * minimun
        maximum = scale * maximum
    x = np.linspace(minimun[0], maximum[0], res)
    y = np.linspace(minimun[1], maximum[1], res)
    z = np.linspace(minimun[2], maximum[2], res)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    X = X.reshape((np.prod(X.shape),))
    Y = Y.reshape((np.prod(Y.shape),))
    Z = Z.reshape((np.prod(Z.shape),))

    points_list = np.column_stack((X, Y, Z))
    del X, Y, Z, x
    return points_list

def get_grid_points():
    mini = [-1.0, -1.0, -1.0]
    maxi = [1.0, 1.0, 1.0]
    grid_points = create_grid_points_from_bounds(mini, maxi, 256)
    grid_points = torch.from_numpy(grid_points).float().to(device)
    grid_points = torch.reshape(grid_points, (1, len(grid_points), 3)).to(device)
    return grid_points

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
                input = torch.cat([points, glob_cond], dim=-1)
                d= deformer(input)
            delta_list.append(d.detach().clone())

            torch.cuda.empty_cache()
        delta = torch.cat(delta_list, dim=1)

    pred_posed = points_neutral[:, :, :3] + delta.squeeze()
    verts = pred_posed.detach().cpu().squeeze().numpy()
    mesh_deformed = trimesh.Trimesh(verts, mesh.faces, process=False)

    

    return mesh_deformed

def shape_interpolation(shape_code1, shape_code2,deform_code, deform_idx):
    interp = torch.linspace(0, 1, 5).to(shape_code1.device)
    shape_code = shape_code1.unsqueeze(0) * (1 - interp).unsqueeze(1) + shape_code2.unsqueeze(0) * interp.unsqueeze(1)
    for i in range(5):
        logits_shape = get_logits(decoder_shape, shape_code[i], get_grid_points())
        mesh_base = mesh_from_logits(logits_shape, [-1, -1, -1], [1, 1, 1], 256)
        deform = deform_mesh(mesh_base, decoder, deform_code)
        save_folder = 'results/cvpr/rebuttal/npm'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        # deform mesh x rotate 180 and z rotate 90
        rot_matrix = get_rotation_matrix(0,90,90)
        deform.apply_transform(rot_matrix)
        deform.export(os.path.join(save_folder, f'deform_{deform_idx}_interp_{i}_rot.obj'))
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RUN Leaf NPM')
    parser.add_argument('--gpu', type=int, default=7, help='gpu index')
    parser.add_argument('--output', type=str, default='shape', help='output directory')
    # setting

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = 'scripts/configs/npm.yaml'
    CFG = yaml.safe_load(open(config, 'r'))


    decoder_shape = UDFNetwork(d_in=CFG['shape_decoder']['decoder_lat_dim'],
                         d_hidden=CFG['shape_decoder']['decoder_hidden_dim'],
                         d_out=CFG['shape_decoder']['decoder_out_dim'],
                         d_in_spatial=3,
                         n_layers=CFG['shape_decoder']['decoder_nlayers'],
                         udf_type='sdf')
    decoder_shape.mlp_pos = None
    decoder = UDFNetwork(d_in=CFG['deform_decoder']['decoder_lat_dim'],
                         d_hidden=CFG['deform_decoder']['decoder_hidden_dim'],
                         d_out=CFG['deform_decoder']['decoder_out_dim'],
                         n_layers=CFG['deform_decoder']['decoder_nlayers'],
                         udf_type='sdf',
                         d_in_spatial=3,
                         geometric_init=False)
   
    # load checkpoint
    checkpoint_shape = torch.load('checkpoints/eccv/shape.tar',map_location=device)
    checkpoint_deform = torch.load('checkpoints/eccv/deform_npm.tar',map_location=device)
    
    latent_shape = checkpoint_shape['latent_idx_state_dict']['weight']
    latent_deform = checkpoint_deform['latent_deform_state_dict']['weight']
    
    decoder_shape.load_state_dict(checkpoint_shape['decoder_state_dict'])
    decoder.load_state_dict(checkpoint_deform['decoder_state_dict'])
    

    decoder = decoder.to(device)
    decoder_shape = decoder_shape.to(device)
    
    # shape interpolation
    for i in range(10):
        # random select deform code
        deform_idx = np.random.randint(0, len(latent_deform))
        shape_interpolation(latent_shape[58], latent_shape[70], latent_deform[deform_idx],deform_idx) # 650