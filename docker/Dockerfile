FROM nvcr.io/nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

ARG USER_ID=1130
ARG GROUP_ID=300
ARG USER_NAME="yyang"
RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime
RUN groupadd -g "${GROUP_ID}" "${USER_NAME}" && useradd -u "${USER_ID}" -m "${USER_NAME}" -g "${USER_NAME}"
RUN echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y libopencv-dev screen git python3.8 python3.8-distutils python3-pip
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
RUN python3.8 -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
# Install PyTorch3D and it's dependencies via Conda
RUN conda install --freeze-installed -c defaults -c fvcore -c iopath -c conda-forge fvcore iopath \
    && git clone https://github.com/facebookresearch/pytorch3d.git \
    && cd pytorch3d \
    && pip install -e .

# Some of our code is using older Jupyter Notebook tooling
# this doesn't work well with some Jupyter Lab bits
RUN conda install -c conda-forge jupyterlab \
    && conda install nodejs \
    && conda install -c defaults -c conda-forge ipympl

# Conda upgrade --all likes to downgrade our numpy install from 1.22 to 1.21
# Because the included PyTorch requires 1.22 to be present to work properly, we need to just force
# reinstall to get 1.22
RUN pip install --force-reinstall numpy==1.22.2

# TODO: Find fix for tensorboard not working on Jupyter Lab 3.x
RUN jupyter labextension update --all \
    && jupyter labextension install @jupyter-widgets/jupyterlab-manager \
    && jupyter labextension install jupyter-matplotlib \
    && jupyter nbextension enable --py widgetsnbextension \
    && jupyter lab build

RUN conda install -c conda-forge pyvista \
    && conda install -c conda-forge ipyvtklink