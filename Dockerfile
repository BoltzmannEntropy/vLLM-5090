FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

# Set working directory
RUN mkdir /root/app
WORKDIR /root/app

# Set non-interactive frontend for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Update and install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    cmake \
    ninja-build \
    gcc \
    g++ \
    build-essential \
    gfortran \
    wget \
    curl \
    pkg-config \
    zip \
    kmod \
    ccache \
    ffmpeg \
    libcudnn9-cuda-12 \
    nodejs \
    npm \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Set environment variables
ARG NPROC=8
ENV MAX_JOBS=16
ENV NVCC_THREADS=4
ENV FLASHINFER_ENABLE_AOT=1
ENV USE_CUDA=1
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST='10.0+PTX'
ENV FLASH_ATTN_CUDA_ARCHS=100
ENV CCACHE_DIR=/root/.ccache
ENV CMAKE_BUILD_TYPE=Release
ENV MAX_JOBS=$NPROC
ENV VLLM_FLASH_ATTN_VERSION=2
ENV TZ=Asia/Dubai

# Ensure pip and python commands use Python 3
RUN ln -s /usr/bin/python3 /usr/bin/python && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone

# Install PyTorch with CUDA 12.8 support
RUN pip install --no-cache-dir torch torchvision torchaudio xformers \
    --index-url https://download.pytorch.org/whl/cu128

# Install additional dependencies
RUN pip install --no-cache-dir \
    qwen-vl-utils accelerate gradio gradio_toggle openai \
    beautifulsoup4 ftfy bitsandbytes datasets optimum auto-gptq \
    soundfile librosa webrtcvad transformers spaces modelscope

# Build vllm from source for NVIDIA 5090 
RUN git clone https://github.com/vllm-project/vllm.git /app

WORKDIR /app/

RUN python3 use_existing_torch.py && \
    pip3 install --no-cache-dir -r requirements/build.txt && \
    pip3 install --no-cache-dir setuptools_scm && \
    pip3 install --no-build-isolation -v -e .

# Install Claude Code globally
RUN npm install -g @anthropic-ai/claude-code

COPY *.sh /app/
COPY *.py /app/

# Clean up
RUN pip3 cache purge

ENV  MKL_SERVICE_FORCE_INTEL=1 
EXPOSE 8097 7842 8501 8000 6666 7860

ENV PYTHONUNBUFFERED=1 	GRADIO_ALLOW_FLAGGING=never 	GRADIO_NUM_PORTS=1 	GRADIO_SERVER_NAME=0.0.0.0     GRADIO_SERVER_PORT=7860 	SYSTEM=spaces



