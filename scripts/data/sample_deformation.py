import trimesh
import numpy as np
import os
# from scripts.dataset.DataManager import LeafScanManager
from multiprocessing import Pool
from matplotlib import pyplot as plt

def sample(m_neutral, m_deformed, std, n_samps):
    p_neutral, idx_neutral = m_neutral.sample(n_samps, return_index=True)
    normals_neutral = m_neutral.face_normals[idx_neutral, :]
    faces = m_neutral.faces[idx_neutral]
    faces_lin = faces.reshape(-1)
    triangles_neutral = m_neutral.vertices[faces_lin].reshape(-1, 3, 3)
    bary = trimesh.triangles.points_to_barycentric(triangles_neutral, p_neutral, method = 'cross')
    offsets = np.random.randn(p_neutral.shape[0]) * std
    offsets = np.expand_dims(offsets, axis=-1)
    p_neutral += offsets * normals_neutral
    
    
    faces = m_deformed.faces[idx_neutral]
    normals= m_deformed.face_normals[idx_neutral, :]
    faces_lin = faces.reshape(-1)
    triangles = m_deformed.vertices[faces_lin, :]
    triangles = triangles.reshape(-1, 3, 3)
    p = trimesh.triangles.barycentric_to_points(triangles, bary)
    p += offsets * normals
    return p_neutral, p, normals_neutral, normals


if __name__ == "__main__":
    VIZ=False
    n_samples = 25000
    #main(n_samples)
    deformation_dir  = 'dataset/deformation_cvpr_new/deform_nonrigid_regis'
    # m_neutral = trimesh.load('dataset/leaf_classification/images/Quercus_Phillyraeoides/43_128.obj' )
    for deform in os.listdir(deformation_dir) :
        if  'rigis' in deform:
            deform_file  = os.path.join(deformation_dir, deform)
            m_deformed = trimesh.load(deform_file)
            base_file = deform_file.replace('rigis', 'base')
            m_neutral = trimesh.load(base_file)
            p_neutral, p, normals_neutral, normals = sample(m_neutral, m_deformed, 0.01, n_samples)
            p_neutral2, p2, normals_neutral2, normals2 = sample(m_deformed, m_neutral, 0.01, n_samples)
            p_neutral, p, normals_neutral, normals = sample(m_neutral, m_deformed, 0.01, n_samples)
            p_neutral2, p2, normals_neutral2, normals2 = sample(m_deformed, m_neutral, 0.01, n_samples)#0.01)
            p_neutral = np.concatenate([p_neutral, p2], axis=0)
            p = np.concatenate([p, p_neutral2], axis=0)
            normals_neutral = np.concatenate([normals_neutral, normals2], axis=0)
            normals = np.concatenate([normals, normals_neutral2], axis=0)

            p_neutral_tight, p_tight, normals_neutral_tight, normals_tight = sample(m_neutral, m_deformed, 0.002, n_samples)#0.002)
            p_neutral_tight2, p_tight2, normals_neutral_tight2, normals_tight2 = sample(m_deformed, m_neutral, 0.002, n_samples)#0.002)
            p_neutral_tight = np.concatenate([p_neutral_tight, p_tight2], axis=0)
            p_tight = np.concatenate([p_tight, p_neutral_tight2], axis=0)
            normals_neutral_tight = np.concatenate([normals_neutral_tight, normals_tight2], axis=0)
            normals_tight = np.concatenate([normals_tight, normals_neutral_tight2], axis=0)          
                                        
            all_p_neutral = np.concatenate([p_neutral, p_neutral_tight], axis=0)
            all_normals_neutral = np.concatenate([normals_neutral, normals_neutral_tight], axis=0)
            all_p = np.concatenate([p, p_tight], axis=0)
            all_normals = np.concatenate([normals, normals_tight], axis=0)
            perm = np.random.permutation(all_p.shape[0])
            all_p_neutral = all_p_neutral[perm, :]
            all_normals_neutral = all_normals_neutral[perm, :]
            all_p = all_p[perm, :]
            all_normals = all_normals[perm, :]
            if np.any(np.isnan(all_p)) or np.any(np.isnan(all_normals)):
                print('DONE')
                break    
            data = np.concatenate([all_p_neutral, all_p], axis=-1)
            data_normals = np.concatenate([all_normals_neutral, all_normals], axis=-1)
            filename = deform.split('/')[-1]
            filename = os.path.splitext(filename)[0] + '_deform.npy'
            np.save(os.path.join(deformation_dir, filename), data)
            print(f'{filename} is saved.' )
                        

    
        