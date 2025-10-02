@REM docker run --gpus all --rm -it --shm-size=8gb --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --memory="16g" --env="DISPLAY"  -p 8000:8000 -p 8078:7842 -p 7861:7860  -p 8502:8501 -v  %cd%/:/root/app -v "c:\Users\user\My Drive\AI-2025\code\":/root/sharedfolder -v c:\Users\User\dev\mdls\.cache\:/root/.cache/  vllm-small-5090:latest

docker run --gpus all --rm -it --shm-size=8gb --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --memory=16g --env=DISPLAY -p 8000:8000 -p 8078:7842 -p 7861:7860 -p 8502:8501 -v "${env:USERPROFILE}/My Drive/AI-2025/:/root/sharedfolder" -v "${env:USERPROFILE}/dev/mdls/.cache:/root/.cache" vllm-small-5090:latest

@REM docker run --gpus all --rm -it --shm-size=8gb --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --memory=16g --env=DISPLAY -p 8000:8000 -p 8078:7842 -p 7861:7860 -p 8502:8501 -v "${env:USERPROFILE}/My Drive/AI-2025/:/root/sharedfolder" -v "${env:USERPROFILE}/dev/mdls/.cache:/root/.cache" registry.hf.space/iqbalzz-hololive-rvc-models-v2:latest

