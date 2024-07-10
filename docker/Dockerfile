# FROM nvcr.io/nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04
FROM nvcr.io/nvidia/pytorch:22.02-py3
ARG USER_ID=1130
ARG GROUP_ID=300
ARG USER_NAME="yyang"
RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime
RUN groupadd -g "${GROUP_ID}" "${USER_NAME}" && useradd -u "${USER_ID}" -m "${USER_NAME}" -g "${USER_NAME}"
RUN echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Get the basic Linux packages upgraded and get the development tools setup
RUN apt-get update \
    && apt-get upgrade -y\
    && apt-get install -y wget \
    build-essential \
    manpages-dev \
    ca-certificates \
    g++ \
    git \
    libglfw3-dev \
    libgles2-mesa-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && gcc --version \
    && conda upgrade --all -y

COPY ./requirements.txt ./tmp/

RUN python3.8 -m pip install -U pip wheel setuptools
RUN python3.8 -m pip install -r ./tmp/requirements.txt
# RUN python3.8 -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
# Install PyTorch3D and it's dependencies via Conda
RUN conda install -c fvcore -c iopath -c conda-forge fvcore iopath \
    && conda install pytorch3d -c pytorch3d
    # && git clone https://github.com/facebookresearch/pytorch3d.git \
    # && cd pytorch3d \
    # && pip install -e .

# Some of our code is using older Jupyter Notebook tooling
# this doesn't work well with some Jupyter Lab bits
RUN conda install -c conda-forge jupyterlab \
    && conda install nodejs \
    && conda install -c defaults -c conda-forge ipympl


RUN conda install -c conda-forge pyvista \
    && conda install -c conda-forge ipyvtklink