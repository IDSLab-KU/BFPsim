# Old pytorch base
# FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime
# New base due to calling cpp code
FROM nvcr.io/nvidia/pytorch:21.03-py3

# Maybe 21.10 will work, too but haven't tested
# FROM nvcr.io/nvidia/pytorch:21.10-py3

# RUN pip install pytorch torchvision

RUN pip install matplotlib

RUN pip install numba==0.53.1

RUN pip install einops

RUN pip install slack_sdk

RUN pip install tensorboard
