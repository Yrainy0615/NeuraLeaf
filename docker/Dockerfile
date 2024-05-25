FROM nvcr.io/nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

ARG USER_ID=1130
ARG GROUP_ID=300
ARG USER_NAME="yyang"
RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime
RUN groupadd -g "${GROUP_ID}" "${USER_NAME}" && useradd -u "${USER_ID}" -m "${USER_NAME}" -g "${USER_NAME}"
RUN echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y libopencv-dev screen git python3.8 python3.8-distutils python3-pip

COPY ./requirements.txt ./tmp/

RUN python3.8 -m pip install -U pip wheel setuptools
RUN python3.8 -m pip install -r ./tmp/requirements.txt
RUN python3.8 -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
