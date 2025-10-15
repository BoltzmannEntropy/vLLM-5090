@echo off
REM Script to run LMCache vLLM container with bash

docker run --gpus all --rm -it ^
    --shm-size=8gb ^
    --ipc=host ^
    --ulimit memlock=-1 ^
    --ulimit stack=67108864 ^
    --memory=16g ^
    --env=DISPLAY ^
    -p 8000:8000 ^
    -p 8078:7842 ^
    -p 7861:7860 ^
    -p 8502:8501 ^
    -v "%USERPROFILE%/dev/:/root/sharedfolder" ^
    -v "%USERPROFILE%/dev/mdls/.cache:/root/.cache" ^
    lmcache/vllm-openai:build-latest ^
    bash
