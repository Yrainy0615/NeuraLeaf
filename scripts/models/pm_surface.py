from geomdl import construct
from geomdl import fitting
from geomdl import exchange
import os
import trimesh
import numpy as np
from scipy.interpolate import griddata
from geomdl import fitting
from geomdl import BSpline
from scipy.spatial import cKDTree
# from geomdl.visualization import VisMPL
def clip_mesh_by_cloud(mesh_verts, mesh_faces, cloud_points, max_dist=0.05):
    """
    根据点云裁剪网格: 如果网格顶点距离点云大于 max_dist, 认为该顶点无效,
    并剔除包含无效顶点的面 (triangle).

    Args:
        mesh_verts: (M, 3) 网格顶点
        mesh_faces: (F, 3) 三角形面(顶点索引)
        cloud_points: (N, 3) 原始点云
        max_dist: 阈值, 距离大于此即视为"不在点云覆盖范围"

    Returns:
        new_verts, new_faces: 剔除后的网格顶点和三角形面 (索引重新排)
    """

    # 1) 建立 k-d 树, 用于最近邻查询
    kdtree = cKDTree(cloud_points)
    
    # 2) 计算每个网格顶点到点云的最近距离
    dists, _ = kdtree.query(mesh_verts)  # shape=(M,)

    # 3) 标记哪些顶点是"无效" (离散网格点距离点云>max_dist)
    invalid_mask = (dists > max_dist)

    # 4) 过滤 faces: 若一个面包含无效顶点 => 丢弃该面
    #    也可以只在“所有顶点都无效”时丢弃，看你的需求
    face_mask = []
    for f in mesh_faces:
        # f is [v0, v1, v2]
        if invalid_mask[f[0]] or invalid_mask[f[1]] or invalid_mask[f[2]]:
            # 任意顶点无效 => 该三角不保留
            face_mask.append(False)
        else:
            face_mask.append(True)
    face_mask = np.array(face_mask, dtype=bool)
    new_faces = mesh_faces[face_mask]
    
    # 5) 只保留仍在使用的顶点, 并重新做索引
    used_vert_ids = np.unique(new_faces)
    # old_id -> new_id
    new_id_map = -1 * np.ones(len(mesh_verts), dtype=int)
    new_id_map[used_vert_ids] = np.arange(len(used_vert_ids))

    # remap 面的顶点索引
    new_faces = new_id_map[new_faces]

    # 提取新的顶点数组
    new_verts = mesh_verts[used_vert_ids]

    return new_verts, new_faces
def approximate_surface_from_scatter(points_3d,
                                     size_u=50,
                                     size_v=50,
                                     degree_u=3,
                                     degree_v=3,
                                     method='linear',
                                     centripetal=False):

    # 1) 拆分 (x, y, z)
    x_vals = points_3d[:, 0]
    y_vals = points_3d[:, 1]
    z_vals = points_3d[:, 2]

    # 2) 找到 bounding box
    x_min, x_max = x_vals.min(), x_vals.max()
    y_min, y_max = y_vals.min(), y_vals.max()

    # 3) 构建规则网格
    # 注意: meshgrid 通常 (rows=size_v, cols=size_u)
    xs = np.linspace(x_min, x_max, size_u)
    ys = np.linspace(y_min, y_max, size_v)
    X, Y = np.meshgrid(xs, ys)  # X,Y shape=(size_v, size_u)

    # 4) 插值: griddata 需要将网格展开成 (M,2) 才能插值
    grid_points_2d = np.column_stack((x_vals, y_vals))  # shape=(N,2)
    query_points_2d = np.column_stack((X.ravel(), Y.ravel()))  # shape=(size_u*size_v,2)

    # griddata插值得到Z
    Zr = griddata(grid_points_2d, z_vals, query_points_2d, method=method)
    # 若有 NaN，可以再次用 nearest 补洞, 或者做其他处理
    mask_nan = np.isnan(Zr)
    if np.any(mask_nan):
        # 简单地用 'nearest' 补
        Zr2 = griddata(grid_points_2d, z_vals, query_points_2d, method='nearest')
        Zr[mask_nan] = Zr2[mask_nan]

    Z = Zr.reshape(Y.shape)  # 重塑回 (size_v, size_u)
    
    points_grid = []
    for i in range(size_v):        # row
        for j in range(size_u):    # col
            xyz = [ X[i,j], Y[i,j], Z[i,j] ]
            points_grid.append(xyz)

    surf = fitting.approximate_surface(
        points_grid, size_v, size_u, degree_v, degree_u, centripetal=centripetal
    )
    # 这样, surf 就是一个 BSpline.Surface 对象
    surf.__reduce_ex__

    return surf, X, Y, Z

# ------------------ 示例使用 ------------------
if __name__ == "__main__":

    deform_root = 'dataset/denseleaf/data_00001/split_regis/'
    save_folder = 'dataset/denseleaf/data_00001/split_regis/fit_ps'    
    for i in os.listdir(deform_root):
        deform_path = os.path.join(deform_root, i)
        file_name = os.path.basename(deform_path)
        points = trimesh.load(deform_path).vertices
        surf, X, Y, Z = approximate_surface_from_scatter(points,
                                                     size_u=100, size_v=100,
                                                     degree_u=3, degree_v=3,
                                                     method='linear',
                                                     centripetal=False)
        print("Created a B-spline surface with", surf.ctrlpts_size_u, "x", surf.ctrlpts_size_v, "control points.")
        save_name = os.path.join(save_folder, file_name).replace('.ply','.obj')
        
        exchange.export_obj(surf, save_name)
        # read this mesh
        mesh = trimesh.load(save_name)
        # cut the mesh by the original point cloud
        mesh.vertices, mesh.faces = clip_mesh_by_cloud(mesh.vertices, mesh.faces, points, max_dist=0.05)
        new_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
        new_mesh.export(save_name.replace('.obj','.ply'))
        print(f"Saved {file_name}")

# if __name__ == "__main__":
#     points = []
#     deform_root = 'dataset/denseleaf/data_00001/split_regis/'
#     save_folder = 'dataset/denseleaf/data_00001/split_regis/fit_ps'
#     os.makedirs(save_folder, exist_ok=True)
#     for i in os.listdir(deform_root):
#         deform_path = os.path.join(deform_root, i)
#         points = trimesh.load(deform_path).vertices
#         surf = ps_fitting(points, method="approximate")
#         exchange.export_obj(surf, os.path.join(save_folder, i.replace('.ply','.obj')))
#         print(f"Saved {i}")
    