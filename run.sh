
# mkdir -p /root/.cache/huggingface/hub
# cp -r /root/sharedfolder/mdls/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct/ ~/.cache/huggingface/hub/


python3 -m vllm.entrypoints.openai.api_server \
    --model google/gemma-3-12b-it \
    --port 8000 \
    --max_model_len 18000 \
    --tensor-parallel-size 1 \
    --gpu_memory_utilization 0.95 \
    --max_num_seqs 1 \
    --enforce-eager


# python3 -m vllm.entrypoints.openai.api_server \
#     --model fancyfeast/llama-joycaption-beta-one-hf-llava \
#     --port 8000 \
#     --max_model_len 24000 \
#     --tensor-parallel-size 1 \
#     --gpu_memory_utilization 0.95 \
#     --max_num_seqs 16 \
#     --enforce-eager

# ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
