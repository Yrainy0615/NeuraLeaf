import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from pytorch3d import transforms
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial import cKDTree as KDTree
import meshlib.mrmeshpy as mm
import meshlib.mrmeshnumpy as mn
from pytorch3d.structures import Meshes
from typing import List

def rts_invert(rts_in):
    """
    rts: ...,3,4   - B ririd transforms
    """
    rts = rts_in.view(-1,3,4).clone()
    Rmat = rts[:,:3,:3] # bs, B, 3,3
    Tmat = rts[:,:3,3:]
    Rmat_i=Rmat.permute(0,2,1)
    Tmat_i=-Rmat_i.matmul(Tmat)
    rts_i = torch.cat([Rmat_i, Tmat_i],-1)
    rts_i = rts_i.view(rts_in.shape)
    return rts_i

def rtk_to_4x4(rtk):
    """
    rtk: ...,12
    """
    device = rtk.device
    bs = rtk.shape[0]
    zero_one = torch.Tensor([[0,0,0,1]]).to(device).repeat(bs,1)

    rmat=rtk[:,:9]
    rmat=rmat.view(-1,3,3)
    tmat=rtk[:,9:12]
    rts = torch.cat([rmat,tmat[...,None]],-1)
    rts = torch.cat([rts,zero_one[:,None]],1)
    return rts

def rtk_compose(rtk1, rtk2):
    """
    rtk ...
    """
    rtk_shape = rtk1.shape
    rtk1 = rtk1.view(-1,12)# ...,12
    rtk2 = rtk2.view(-1,12)# ...,12

    rts1 = rtk_to_4x4(rtk1)
    rts2 = rtk_to_4x4(rtk2)

    rts = rts1.matmul(rts2)
    rvec = rts[...,:3,:3].reshape(-1,9)
    tvec = rts[...,:3,3].reshape(-1,3)
    rtk = torch.cat([rvec,tvec],-1).view(rtk_shape)
    return rtk

def vec_to_sim3(vec):
    """
    vec:      ...,10
    center:   ...,3
    orient:   ...,3,3
    scale:    ...,3
    """
    center = vec[...,:3]
    orient = vec[...,3:7] # real first
    orient = F.normalize(orient, 2,-1)
    orient = transforms.quaternion_to_matrix(orient) # real first
    scale =  vec[...,7:10].exp()
    return center, orient, scale

def bone_transform(bones_in, rts, is_vec=False):
    """ 
    bones_in: 1,B,10  - B gaussian ellipsoids of bone coordinates
    rts: ...,B,3,4    - B ririd transforms
    rts are applied to bone coordinate transforms (left multiply)
    is_vec:     whether rts are stored as r1...9,t1...3 vector form
    """
    B = bones_in.shape[-2]
    bones = bones_in.view(-1,B,10).clone()
    if is_vec:
        rts = rts.view(-1,B,12)
    else:
        rts = rts.view(-1,B,3,4)
    bs = rts.shape[0] 

    center = bones[:,:,:3]
    orient = bones[:,:,3:7] # real first
    scale =  bones[:,:,7:10]
    if is_vec:
        Rmat = rts[:,:,:9].view(-1,B,3,3)
        Tmat = rts[:,:,9:12].view(-1,B,3,1)
    else:
        Rmat = rts[:,:,:3,:3]   
        Tmat = rts[:,:,:3,3:4]   

    # move bone coordinates (left multiply)
    center = Rmat.matmul(center[...,None])[...,0]+Tmat[...,0]
    Rquat = transforms.matrix_to_quaternion(Rmat)
    orient = transforms.quaternion_multiply(Rquat, orient)

    scale = scale.repeat(bs,1,1)
    bones = torch.cat([center,orient,scale],-1)
    return bones 

def rtmat_invert(Rmat, Tmat):
    """
    Rmat: ...,3,3   - rotations
    Tmat: ...,3   - translations
    """
    rts = torch.cat([Rmat, Tmat[...,None]],-1)
    rts_i = rts_invert(rts)
    Rmat_i = rts_i[...,:3,:3] # bs, B, 3,3
    Tmat_i = rts_i[...,:3,3]
    return Rmat_i, Tmat_i

def axis_rotate(orient, mdis):
    bs,N,B,_,_ = mdis.shape
    mdis = (orient * mdis.view(bs,N,B,1,3)).sum(4)[...,None] # faster 
    #mdis = orient.matmul(mdis) # bs,N,B,3,1 # slower
    return mdis

def vis_points(points:torch.tensor):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = points[:, 0].numpy()
    y = points[:, 1].numpy()
    z = points[:, 2].numpy()
    ax.scatter(x, y, z)
    ax.set_xlabel('X Coordinates')
    ax.set_ylabel('Y Coordinates')
    ax.set_zlabel('Z Coordinates')
    # save the figure
    plt.savefig('bone.png')
    plt.close()


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


def grid_to_occ(grid,pts,res=128):
    # Convert grid to numpy if it's a torch tensor
    if isinstance(grid, torch.Tensor):
        grid = grid.cpu().numpy()
    # Convert points to numpy if it's a torch tensor
    if isinstance(pts, torch.Tensor):
        pts = pts.cpu().numpy()
    
    kdtree = KDTree(grid)
    occupancies = np.zeros(len(grid), dtype=np.int8)
    _, idx = kdtree.query(pts)
    occupancies[idx] = 1
    occupancy_grid = occupancies.reshape(res,res,res)
    return occupancy_grid


def raw_to_canonical(points):
    points = points - points.mean(0)
    scale = torch.max(torch.abs(points))
    if scale > 0:
        points = points / scale
    return points


def points_to_voxel(points, resolution=128):
    points = raw_to_canonical(points)
    grid = create_grid_points_from_bounds([0, 0, 0], [1, 1, 1], resolution)
    grid = torch.tensor(grid, dtype=torch.float32)
    voxel = grid_to_occ(grid, points, res=resolution)
    return voxel


def mask2mesh(masks:List[torch.Tensor]):
    """Convert mask list to mesh list."""
    meshes = []
    for mask in masks:
        mask = mask.squeeze(0).detach().cpu().numpy()
        y_indices, x_indices = np.where(mask == 1)
        z_coords = np.zeros_like(x_indices)
        u_coords = x_indices / mask.shape[1]
        v_coords = y_indices / mask.shape[0]
        verts = np.stack([u_coords, v_coords, z_coords], axis=-1)
        # Random sample 15000 points
        if verts.shape[0] > 15000:
            idx = np.random.choice(verts.shape[0], 15000, replace=False)
            verts = verts[idx]
        pc = mn.pointCloudFromPoints(verts)
        pc.invalidateCaches()
        mesh = mm.triangulatePointCloud(pc)
        verts = mn.getNumpyVerts(mesh)
        faces = mn.getNumpyFaces(mesh.topology)
        mesh_torch3d = Meshes(verts=[torch.tensor(verts).float()], faces=[torch.tensor(faces).long()])
        meshes.append(mesh_torch3d)
    return meshes[0] if len(meshes) == 1 else meshes


