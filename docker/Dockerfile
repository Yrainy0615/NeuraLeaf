FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ARG USER_ID=1130
ARG GROUP_ID=300
ARG USER_NAME="yyang"

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime && \
    groupadd -g "${GROUP_ID}" "${USER_NAME}" && \
    useradd -u "${USER_ID}" -m "${USER_NAME}" -g "${USER_NAME}" && \
    echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

ENV DEBIAN_FRONTEND=noninteractive

# basic tools
RUN apt-get update && apt-get install -y \
    build-essential wget git vim cmake \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

############################
# install Miniconda + mamba  #
############################

ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniconda.sh
ENV PATH=${CONDA_DIR}/bin:${PATH}

# Accept conda Terms of Service
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# install mamba, accelerate environment creation
RUN conda install -n base -c conda-forge mamba -y && conda clean -afy

###########################
# create neuraleaf env    #
###########################

ARG ENV_NAME=neuraleaf

# basic env
RUN mamba create -n ${ENV_NAME} python=3.10 -y && conda clean -afy

# Install PyTorch + CUDA 11.8
RUN conda run -n ${ENV_NAME} pip install \
    torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118

# Install numpy and other scientific packages via conda to avoid compilation issues
RUN mamba install -n ${ENV_NAME} -c conda-forge \
    numpy \
    scipy \
    scikit-learn \
    -y && conda clean -afy

# Install other packages from requirements.txt
COPY requirements.txt /tmp/requirements.txt
RUN conda run -n ${ENV_NAME} pip install -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# use bash
SHELL ["bash", "-lc"]

# write activate command for root and user
RUN echo "conda activate ${ENV_NAME}" >> /root/.bashrc
RUN echo "conda activate ${ENV_NAME}" >> /home/${USER_NAME}/.bashrc

USER ${USER_NAME}
WORKDIR /home/${USER_NAME}/mnt/workspace
