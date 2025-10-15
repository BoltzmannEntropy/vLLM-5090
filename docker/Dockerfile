# The LMCache Dockerfile is used to build a LMCache image that is integrated
# to run with vLLM OpenAI server.

# Please update any changes made here to
# docs/source/developer_guide/docker_file.rst
# docs/source/getting_started/installation.rst
# docs/production/docker_deployment.rst

ARG CUDA_VERSION=12.8
ARG UBUNTU_VERSION=24.04

#################### BASE BUILD IMAGE ####################
# Prepare basic build environment

FROM nvcr.io/nvidia/cuda-dl-base:25.03-cuda${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS base

ARG CUDA_VERSION
ARG PYTHON_VERSION=3.12
ARG UBUNTU_VERSION
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and other dependencies
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update -y \
    && apt-get install -y --no-install-recommends \
        ccache software-properties-common git curl sudo \
        python3 python3-dev python3-venv python3-pip tzdata \
        nodejs npm \
    && ldconfig /usr/local/cuda-$(echo $CUDA_VERSION | cut -d. -f1,2)/compat/ \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv ~/.local/bin/uv /usr/local/bin/ \
    && mv ~/.local/bin/uvx /usr/local/bin/ \
    && rm -f /usr/lib/python*/EXTERNALLY-MANAGED \
    && python3 --version \
    && node --version \
    && npm --version

WORKDIR /workspace

# Install global npm packages for AI/code tools
RUN npm install -g opencode-ai @anthropic-ai/claude-code typescript ts-node

# Copy OpenCode configuration
COPY opencode.json /workspace/opencode.json
RUN mkdir -p /root/.config/opencode && \
    cp /workspace/opencode.json /root/.config/opencode/opencode.json

# Install runtime dependencies
COPY ./requirements/common.txt common.txt
COPY ./requirements/cuda.txt cuda.txt
COPY ./requirements/ds-stack.txt ds-stack.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    uv pip install --system -r cuda.txt && \
    uv pip install --system -r ds-stack.txt

# CUDA arch list used by torch
ARG torch_cuda_arch_list='10.0+PTX 12.0+PTX'
ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}

# Override the arch list for flash-attn to reduce the binary size
ARG vllm_fa_cmake_gpu_arches='80-real;90-real;100-real'
ENV VLLM_FA_CMAKE_GPU_ARCHES=${vllm_fa_cmake_gpu_arches}
ENV FLASH_ATTN_CUDA_ARCHS=120

#################### vLLM IMAGE & LMCache (Build) ##########################
# Integrate vLLM nightly build and LMCache build, and expose vLLM OpenAI API

FROM base AS image-build

# install build dependencies
COPY ./requirements/build.txt build.txt

# Max jobs used by Ninja to build extensions
ARG max_jobs=2
ENV MAX_JOBS=${max_jobs}

# Number of threads used by nvcc
ARG nvcc_threads=8
ENV NVCC_THREADS=$nvcc_threads

ARG VLLM_VERSION=nightly

RUN --mount=type=cache,target=/root/.cache/pip \
    uv pip install --system -r build.txt

ARG LMCACHE_COMMIT_ID=1

COPY . /workspace/LMCache
WORKDIR /workspace/LMCache

# Set version for setuptools-scm since we don't have git metadata in the build context
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE=0.1.0

# Build LMCache.
# As we are installing vLLM before LMCache, we will implicitly use
# the same torch version as the vLLM build when running the LMCache build.
# This means that LMCache is in sync with the vLLM torch version.
RUN --mount=type=cache,target=/root/.cache/ccache,id=ccache \
    --mount=type=cache,target=/root/.cache/uv,id=uv-cache,sharing=locked \
    if [ "$VLLM_VERSION" = "nightly" ]; then \
        uv pip install --system --prerelease=allow 'vllm[runai,tensorizer,flashinfer]' \
            --extra-index-url https://wheels.vllm.ai/nightly \
            --index-strategy unsafe-best-match ; \
    else \
        uv pip install --system --prerelease=allow "vllm[runai,tensorizer,flashinfer]==${VLLM_VERSION}" ; \
    fi && \
    python3 -c 'import torch; print("TORCH=", torch.__version__)' && \
    python3 setup.py bdist_wheel --dist-dir=dist_lmcache && \
    uv pip install --system ./dist_lmcache/*.whl --verbose

WORKDIR /workspace

# Copy app files to workspace for easy access
COPY app-*.py /workspace/

# Copy scripts directory for training and inference
COPY scripts/ /workspace/scripts/

# ENTRYPOINT ["vllm", "serve"]

#################### vLLM IMAGE & LMCache (Release) #######################
# Integrate vLLM and LMCache stable releases, and expose vLLM OpenAI API

FROM base AS image-release

# Install LMCache and vLLM stable releases.
# It is imperative that LMCache uses the same torch version as the
# vLLM stable release.
RUN uv pip install --system --prerelease=allow vllm[runai,tensorizer,flashinfer] && \
    uv pip install --system lmcache --verbose

# Install data science stack for training and inference
COPY ./requirements/ds-stack.txt ds-stack.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    uv pip install --system -r ds-stack.txt

# Copy scripts directory for training and inference
COPY scripts/ /workspace/scripts/

WORKDIR /workspace
# ENTRYPOINT ["vllm", "serve"]
