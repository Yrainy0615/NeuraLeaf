FROM nvcr.io/nvidia/cuda:11.6.1-cudnn8-runtime-ubuntu20.04
# FROM nvcr.io/nvidia/pytorch:22.02-py3
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
    && gcc --version 

# install and setup conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && bash ~/miniconda.sh -b -p /opt/miniconda \
    && rm ~/miniconda.sh
ENV PATH="/opt/miniconda/bin:$PATH"

# create conda environment
RUN conda init bash \
    && conda create -n neuraleaf python=3.9 -y \
    && echo "conda activate neuraleaf" >> ~/.bashrc
# install pytorch3d by conda
COPY ./requirements.txt ./tmp/
RUN /bin/bash -c ". activate neuraleaf && \
    conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia -y && \
    conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y  && \
    pip install -r ./tmp/requirements.txt"

# RUN python3.8 -m pip install -U pip wheel setuptools

