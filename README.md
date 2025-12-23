# NeuraLeaf

Implementation of NeuraLeaf - a neural representation for 3D leaf reconstruction and generation.

[Paper](https://arxiv.org/pdf/2507.12714) | [Project Page](https://neuraleaf-yang.github.io/)

<div align="center">
  <img src="assets/teaser_v1.png" alt="NeuraLeaf Teaser" width="800"/>
</div>

## Dataset

### DeformLeaf Dataset

We have constructed a unique dataset of **3D deformed leaves** paired with their corresponding **2D base shapes** containing RGB, Mask, and Mesh.  
You can download the full dataset here:  
[🔗 DeformLeaf Dataset](https://drive.google.com/drive/folders/1qW-Q5zg-JS9n2p9KOyQnPcq3UJfJdCx5)

### Dataset Format

The dataset should be organized as follows:

```
/mnt/data/cvpr_final/
├── deform_train/          # Deformed mesh files
│   ├── leaf_1_deform.ply  # Deformed leaf meshes (.obj or .ply)
│   ├── leaf_2_deform.ply
│   └── ...
├── base_shape/            # Base shape mesh files
│   ├── leaf_1.obj         # Base shape meshes (.obj)
│   ├── leaf_2.obj
│   └── ...
└── base_mask/            # Base shape masks
    ├── leaf_1.png         # 2D base masks (.png)
    ├── leaf_2.png
    └── ...
    └── sdf/               # Optional: SDF files
        ├── leaf_1_sdf.npy
        └── ...
```

**File Naming Convention:**
- Deformed meshes: `{base_name}_deform.ply` or `{base_name}.obj`
- Base shapes: `{base_name}.obj`
- Base masks: `{base_name}.png`
- Files are matched by `base_name` (without extension)

**Data Structure:**
- `DeformLeafDataset` loads pairs of (deformed mesh, base shape, base mask)
- Each sample contains:
  - `deform_points`: Point cloud from deformed mesh (N, 3)
  - `base_mesh`: Path to base shape mesh file
  - `base_name`: Base name for matching
  - `shape_idx`: Index to corresponding base shape in BaseShapeDataset
  - `idx`: Index of the deformation sample

## Installation

You can install NeuraLeaf using either Docker or manual conda installation.

### option 1: Docker Installation
We provide a Dockerfile for building environment.

### Option 2: Conda Installation

#### Step 1: Create Conda Environment

```bash
conda create -n neuraleaf python=3.9 -y
conda activate neuraleaf
```

#### Step 2: Install PyTorch (CUDA 11.8)

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118
```

#### Step 3: Install PyTorch3D

```bash
conda install -y -c pytorch3d -c pytorch -c fvcore -c iopath -c bottler -c conda-forge \
    fvcore iopath nvidiacub pytorch3d
```

#### Step 4: Install Other Dependencies

```bash
# Install conda packages
conda install -y -c conda-forge scipy scikit-learn

# Install pip requirements
pip install -r requirements.txt
```

## Training Scripts

### 1. `scripts/train/train_2dsdf.py`

**Purpose:** Train the 2D SDF/base shape decoder.

This script trains the shape decoder that converts shape latent codes to 2D SDF fields, which are then converted to binary masks representing the base leaf shape.

**Usage:**
```bash
python scripts/train/train_2dsdf.py \
    --config scripts/configs/bashshape.yaml \
```

**Key Features:**
- Trains `UDFNetwork` decoder to predict 2D SDF from shape codes
- Learns shape latent codes via embedding layer
- Optimizes temperature parameter `k` for mask generation

### 2. `scripts/train/train_deform_w_shape.py`

**Purpose:** Train the deformation model with shape prior.

This script trains the deformation network that learns to deform base shapes into various poses. It uses:
- Shape codes (from pretrained shape decoder)
- Deform codes (learned embeddings)
- Skinning weights predictor (SWPredictor)
- Transformation predictor (TransPredictor)
- Global bone positions

**Usage:**
```bash
python scripts/train/train_deform_w_shape.py \
    --config scripts/configs/deform.yaml
```

**Key Features:**
- Jointly optimizes deform codes, bone positions, skinning weights, and transformations
- Supports other shape regularizers like ARAP and length regularization

### 3. `scripts/train/train_encoder.py`

**Purpose:** Train encoders to predict initial shape and deform codes from input point clouds.

This script trains encoders that take point clouds (converted to SDF grids) as input and predict the corresponding shape and deformation codes. The training uses ground truth codes from pretrained models.

**Usage:**
```bash
python scripts/train/train_encoder.py \
    --config scripts/configs/deform.yaml \
    --shape_checkpoint checkpoints/baseshape_l.pth \
    --deform_checkpoint checkpoints/deform.pth \
```

**Key Features:**
- Converts point clouds to SDF grids (voxel representation)
- Trains separate encoders for shape and deformation codes

## Fitting

The `fitting.py` script fits a given mesh to the learned deformation model by optimizing shape and deformation codes.

### Usage

```bash
python fitting.py \
    --mesh_path /path/to/input \
    --method neuraleaf \
    --save_folder /path/to/save_folder
```

### Parameters

**Required:**
- `--mesh_path`: Path to input mesh/point_cloud file (.obj or .ply)
- `--save_folder`: Output directory for fitted meshes

**Optional:**
- `--method`: Fitting method
  - `neuraleaf`: Optimize shape and deform codes. This method works well when the input mesh is within the learned parameter space.
  - `direct`: Directly optimize skinning weights, bones, and transformations. Since NeuraLeaf is a parametric model, it may struggle with inputs that fall outside the training distribution. When the input mesh exceeds the parameter range or is significantly different from the training data, you can try the `direct` method as a fallback, which bypasses the learned latent codes and directly optimizes the deformation parameters. 
- `--use_length_reg`: Enable length regularization to preserve boundary edge lengths
- `--use_arap`: Enable ARAP (As-Rigid-As-Possible) loss for shape regularization

### Output

The script generates:
- `{mesh_name}_fitted.obj`: Fitted deformed mesh
- `{mesh_name}_fitted_base.obj`: Base shape mesh
- `{mesh_name}_fitted_deform.obj`: Deformed point cloud (if applicable)

## Generation

The `generation.py` script provides three generation functions for creating new leaf meshes.

### Usage

```bash
python generation.py \
    --num_deforms_per_shape 5 \
    --mode random
```

### Generation Functions

#### 1. Random Generation
Generate a single mesh from random or specified shape and deform codes.

```python
from generation import Generator

generator = Generator(...)
mesh = generator.random_generation(shape_idx=0, deform_idx=5)
# Returns: Single PyTorch3D Meshes object
```

#### 2. Shape Interpolation
Interpolate between two shape codes with a fixed deform code.

```python
meshes = generator.shape_interpolation(
    shape_idx1=0, 
    shape_idx2=10, 
    deform_idx=5, 
    n_steps=5
)
# Returns: List of 5 meshes showing shape transition
```

#### 3. Deformation Interpolation
Interpolate between two deform codes with a fixed shape code.

```python
meshes = generator.deformation_interpolation(
    shape_idx=0, 
    deform_idx1=5, 
    deform_idx2=15, 
    n_steps=5
)
# Returns: List of 5 meshes showing deformation transition
```

### Parameters

**Required:**
- `--shape_checkpoint`: Path to pretrained shape model checkpoint
- `--deform_checkpoint`: Path to pretrained deformation model checkpoint

**Optional:**
- `--config`: Shape config file (default: `scripts/configs/bashshape.yaml`)
- `--config_deform`: Deform config file (default: `scripts/configs/deform.yaml`)
- `--output_dir`: Output directory for generated meshes 
- `--num_deforms_per_shape`: Number of random deform codes per shape generation (default: 5)

### Output Format

When generating encoder training dataset, meshes are saved as:
- `{shape_idx}_{deform_idx}.obj`: Generated mesh with shape_idx and deform_idx

## Pretrained Weights

Download pretrained model checkpoints from: [Google Drive](https://drive.google.com/drive/folders/1v9K7HJJVlCXkPfgXuEumLp9AP8iIS14G?usp=drive_link)

The download includes the following files:
- `baseshape.pth` - Shape model checkpoint
- `deform.pth` - Deformation model checkpoint
- `encoder.pth` - Shape encoder checkpoint
- `baseshape_l.pth` - Larger baseshape space checkpoint (default)

After downloading, place the checkpoints in the following directory structure:
```
checkpoints/
├── baseshape.pth
├── deform.pth
├── encoder.pth
└── baseshape_l.pth
```

## Citation

If you find our code, dataset or paper useful, please consider citing

```bibtex
@InProceedings{Yang_2025_ICCV,
    author    = {Yang, Yang and Mao, Dongni and Santo, Hiroaki and Matsushita, Yasuyuki and Okura, Fumio},
    title     = {NeuraLeaf: Neural Parametric Leaf Models with Shape and Deformation Disentanglement},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year      = {2025},
}
```



