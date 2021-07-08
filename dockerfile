FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

RUN pip install matplotlib

RUN pip install numba

RUN pip install einops