# FROM nvcr.io/nvidia/pytorch:24.10-py3
FROM nvcr.io/nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    git \
    curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

SHELL [ "/bin/bash", "-c" ]

RUN git clone https://github.com/NVlabs/VILA.git -b LongVILA

COPY vila_setup.sh /app/vila_setup.sh

COPY remembr /app/remembr
COPY requirements.txt /app/requirements.txt
COPY setup.py /app/setup.py
COPY examples /app/examples

# Install Miniconda x86_64
RUN curl -LO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

ENV PATH="/opt/conda/bin:$PATH"

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda init bash && \
    conda create -n remembr python=3.10

# Install ros2
# pip install opencv-python-headless
