@echo off
REM Example script to build the LMCache integrated with vLLM container image

REM Update the following variables accordingly
set CUDA_VERSION=12.8
set DOCKERFILE_NAME=Dockerfile
set VLLM_VERSION=nightly
set DOCKER_BUILD_PATH=.\
set UBUNTU_VERSION=24.04

REM `image-build` target will use the latest LMCache and vLLM code
REM Change to 'image-release' target for using release package versions of vLLM and LMCache
set BUILD_TARGET=image-build

set IMAGE_TAG=lmcache/vllm-openai:build-latest

echo Building Docker image with full data science stack...
echo This will take 15-20 minutes...

docker build ^
    --build-arg CUDA_VERSION=%CUDA_VERSION% ^
    --build-arg UBUNTU_VERSION=%UBUNTU_VERSION% ^
    --build-arg VLLM_VERSION=%VLLM_VERSION% ^
    --target %BUILD_TARGET% --file docker\%DOCKERFILE_NAME% ^
    --tag %IMAGE_TAG% %DOCKER_BUILD_PATH%

echo.
echo Build complete! Run with: run-d.bat
echo Image: %IMAGE_TAG%
