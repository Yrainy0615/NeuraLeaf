import os
import sys
sys.path.append('scripts')
from utils.utils import mask_to_mesh
from meshlib import mrmeshpy
import trimesh

class data_processor():
    def __init__(self, root_dir):
        self.all_mask = []
        self.all_mask_with_rgb = []
        self.all_mask_wo_rgb = []
        self.all_rgb = []
        self.all_sdf = []
        self.all_base_mesh = []
        self.all_deformed_mesh = None
        for root, dirs, files in os.walk(os.path.join(root_dir,'LeafData')):
            for file in files:
                if file.endswith('.npy'):
                    self.all_sdf.append(os.path.join(root, file))
                elif file.endswith('.ply'):
                    self.all_base_mesh.append(os.path.join(root, file))
                elif file.endswith('.JPG'):
                    if 'mask' in file and not 'sdf' in file:
                        self.all_mask_with_rgb.append(os.path.join(root, file))
                    elif not 'mask' in file:
                        self.all_rgb.append(os.path.join(root, file)) 
                elif file.endswith('.jpg'):
                    self.all_mask_wo_rgb.append(os.path.join(root, file))
                elif file.endswith('ply'):
                    self.all_base_mesh.append(os.path.join(root, file))
        self.all_mask = self.all_mask_with_rgb + self.all_mask_wo_rgb
        # sort all
        self.all_mask.sort()
        self.all_mask_with_rgb.sort()
        self.all_mask_wo_rgb.sort()
        self.all_rgb.sort()
        self.all_sdf.sort()
        self.all_base_mesh.sort()

    

    
    def sample_deformation(self):
        pass
        

if __name__ == "__main__":
    root = 'dataset/LeafData'
    processor = data_processor(root)
    
    pass
      