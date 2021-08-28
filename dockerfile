FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

RUN pip install matplotlib

RUN pip install numba

RUN pip install einops

RUN pip install slack_sdk

RUN pip install tensorboard