import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from pytorch3d import transforms
from matplotlib import pyplot as plt

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
    