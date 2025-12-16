FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install basic tools and OpenCV dependencies
RUN apt-get update && apt-get install -y \
    build-essential wget git vim cmake \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    libopengl0 libglu1-mesa mesa-utils \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

############################
# Miniconda
############################

ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniconda.sh
ENV PATH=${CONDA_DIR}/bin:${PATH}

# Accept conda ToS
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN conda install -n base -c conda-forge mamba -y && conda clean -afy

ARG ENV_NAME=neuraleaf

# Create Python environment
RUN mamba create -n ${ENV_NAME} python=3.9 -y && conda clean -afy

# Install PyTorch (CUDA 11.8 compatible)
RUN conda run -n ${ENV_NAME} pip install \
    torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu117
# install pytorch3d 
RUN conda install -y -n ${ENV_NAME} \
    -c pytorch3d -c pytorch -c fvcore -c iopath -c bottler -c conda-forge \
    fvcore iopath nvidiacub pytorch3d
# Install other conda packages
RUN mamba install -n ${ENV_NAME} -c conda-forge \
    scipy \
    scikit-learn \
    -y && conda clean -afy

# Install pip requirements
COPY requirements.txt /tmp/requirements.txt
RUN conda run -n ${ENV_NAME} pip install -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

SHELL ["bash", "-lc"]

# Setup conda activation in bashrc
RUN echo "source ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate ${ENV_NAME}" >> /root/.bashrc

WORKDIR /workspace

