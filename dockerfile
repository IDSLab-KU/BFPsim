FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

# FROM nvcr.io/nvidia/pytorch:21.10-py3
# FROM nvcr.io/nvidia/pytorch:19.12-py3
FROM nvcr.io/nvidia/pytorch:21.03-py3
# FROM nvcr.io/nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu18.04

# RUN pip install pytorch torchvision

RUN pip install matplotlib

RUN pip install numba==0.53.1

RUN pip install einops

RUN pip install slack_sdk

RUN pip install tensorboard

# RUN apt-get update && \
#     apt-get -y install gcc && \
#     rm -rf /var/lib/apt/lists/*