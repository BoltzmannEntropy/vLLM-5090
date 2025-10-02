import os
import base64
import io
import pandas as pd
from PIL import Image
import torch
from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
import gradio as gr
from tqdm import tqdm
import gc

# GPU configuration
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Clean GPU memory utility
def clean_memory():
    """Utility to clean GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()

# Define constants for models
available_models = [
    "Qwen/Qwen2-VL-7B-Instruct",
    'Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4',
]

# Prompt for person detection
person_detection_prompt = (
    "Detect and describe the person in the image. Focus on identifying the person's posture, clothing, and any visible accessories. "
    "Your response should include details such as the person's position in the image, clothing color, and any notable items they are carrying."
)

def encode_image(image):
    img_buffer = io.BytesIO()
    image.save(img_buffer, format="PNG")
    img_str = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
    return f'<img src="data:image/png;base64,{img_str}" style="max-width:500px;"/>'

def generate_output(model, prompt, image, model_name):
    try:
        # Prepare the prompt based on the model
        if model_name == "llava-hf/llava-1.5-7b-hf":
            prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
        elif model_name == "llava-hf/llava-v1.6-mistral-7b-hf":
            prompt = f"[INST] <image>\n{prompt} [/INST]"
        elif model_name == "llava-hf/LLaVA-NeXT-Video-7B-hf":
            prompt = f"USER: <video>\n{prompt} ASSISTANT:"
        elif model_name == "adept/fuyu-8b":
            prompt = f"{prompt}\n"
        elif model_name == "microsoft/Phi-3-vision-128k-instruct":
            prompt = f"<|user|>\n<|image_1|>\n{prompt}<|end|>\n<|assistant|>\n"
        elif model_name == "google/paligemma-3b-mix-224":
            prompt = "caption en"
        elif model_name == "facebook/chameleon-7b":
            prompt = f"{prompt}<image>"
        elif model_name == "openbmb/MiniCPM-V-2_6":
            prompt = f"(<image>./</image>)\n{prompt}"
        elif model_name == "OpenGVLab/InternVL2-2B":
            prompt = f"<image>\n{prompt}"
        elif model_name == "Salesforce/blip2-opt-2.7b":
            prompt = f"Question: {prompt} Answer:"
        elif model_name == "Qwen/Qwen-VL":
            prompt = f"{prompt}Picture 1: <img></img>\n"
        elif model_name in ["Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4",
                            "Qwen/Qwen2-VL-7B-Instruct", 
                            "Qwen/Qwen2-VL-72B-Instruct",
                            "Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4",
                            "Qwen/Qwen2-VL-72B"]:
            prompt = (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                f"{prompt}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

        # Prepare the input for vLLM
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {
                "image": image
            },
        }

        # Generate output
        sampling_params = SamplingParams(max_tokens=1024, temperature=0.7, top_p=0.9)
        outputs = model.generate(inputs, sampling_params=sampling_params)
        return outputs[0].outputs[0].text

    except Exception as e:
        return f"Error during generation: {str(e)}"

def process_images_from_root_folder(root_folder, models, prompt, output_csv_path, progress=None, device_map="auto", torch_dtype=torch.float16):
    # Initialize df to an empty DataFrame
    df = pd.DataFrame()

    # Convert single model to list if needed
    if isinstance(models, str):
        models = [models]

    # Validate models
    for model_name in models:
        if model_name not in available_models:
            raise ValueError(f"Invalid model name: {model_name}. Available models: {available_models}")

    data = []
    images = []

    # Walk through the root folder
    for root, dirs, files in os.walk(root_folder):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, filename)
                try:
                    img = Image.open(image_path).convert("RGB")
                    images.append({
                        "filename": filename,
                        "image": img,
                        "folder_path": root
                    })
                except Exception as e:
                    print(f"Error opening image {filename}: {e}")

    model_processors = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load models
    for model_name in models:
        try:
            print(f"Loading model: {model_name}")
            clean_memory()

            # model = LLM(
            #     model=model_name,
            #     max_num_seqs=5,
            # )
            llm = LLM(
                #dtype="half",                -> Changed this
                #max_num_seqs=128,            -> Changed this
                model=model_name,
                # dtype = torch.bfloat16, 
                max_num_seqs = 16,
                max_model_len=4096,#4096*5,         
                trust_remote_code=True,     
                # tensor_parallel_size=4,      
                # gpu_memory_utilization=0.96, 
            )

            # llm = LLM(
            #     dtype="half",                # The data type for the model weights and activations
            #     max_num_seqs=8,              # Maximum number of sequences per iteration. Default is 256
            #     max_model_len=4096,          # Model context length
            #     trust_remote_code=True,      # Trust remote code (e.g., from HuggingFace) when downloading the model and tokenizer
            #     tensor_parallel_size=4,      # The number of GPUs to use for distributed execution with tensor parallelism
            #     gpu_memory_utilization=0.97, # The ratio (between 0 and 1) of GPU memory to reserve for the model
            # )

            model_processors[model_name] = llm
            print(f"Successfully loaded model: {model_name}")

        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            continue

    if not model_processors:
        raise RuntimeError("No models were successfully loaded")

    for idx, img in enumerate(tqdm(images, desc="Processing images")):
        print(f"\nProcessing: {os.path.join(img['folder_path'], img['filename'])}")

        # Initialize row data
        row_data = {
            "Image_Name": img['filename'],
            "Description": ""
        }

        for model_name in models:
            if model_name not in model_processors:
                continue

            model = model_processors[model_name]

            try:
                response_text = generate_output(model, prompt, img['image'], model_name)
                print(f"Model {model_name} response: {response_text}")

                # Store the description
                model_prefix = model_name.split('/')[-1]
                row_data[f"{model_prefix}_Description"] = response_text

            except Exception as e:
                print(f"Error processing with model {model_name}: {e}")
                model_prefix = model_name.split('/')[-1]
                row_data[f"{model_prefix}_Description"] = "ERROR"
        
        data.append(row_data)
        
        # Save progress after each image
        df = pd.DataFrame(data)
        df.to_csv(output_csv_path, index=False)
        print(f"Progress saved to {output_csv_path}")

        # Update progress
        if progress is not None:
            progress((idx + 1) / len(images), desc=f"Processed {idx + 1}/{len(images)} images")

    print(f"\nProcessing complete. Final results written to {output_csv_path}")
    return df, output_csv_path

def gradio_interface(root_folder, models, output_csv_path, device_map, torch_dtype_str, progress=gr.Progress()):
    # Convert torch_dtype string to actual torch.dtype
    torch_dtype = torch.float16 if torch_dtype_str == "torch.float16" else torch.float32

    df, csv_path = process_images_from_root_folder(root_folder, models, person_detection_prompt, output_csv_path, progress, device_map, torch_dtype)
    return df.to_html(classes='table table-striped', index=False), csv_path

if __name__ == "__main__":
    with gr.Blocks() as demo:
        gr.Markdown("# vLLM - Person Detection")
        gr.Markdown(f"**Prompt:** {person_detection_prompt}")

        with gr.Row():
            root_folder = gr.Textbox(label="Root Folder", value=os.path.expanduser("~/app/images/"))
            models = gr.CheckboxGroup(label="Models", choices=available_models, value=[available_models[0]])
            output_csv_path = gr.Textbox(label="Output CSV Path", value="person_detection.csv")
        with gr.Row():
            device_map = gr.Radio(choices=["auto", "cpu", "cuda"], label="Device Map", value="auto")
            torch_dtype = gr.Radio(choices=["torch.float16", "torch.float32"], label="Torch Dtype", value="torch.float16")

        submit_button = gr.Button("Process Images")

        with gr.Tabs():
            with gr.Tab("Results"):
                output_html = gr.HTML()

        download_button = gr.File(label="Download CSV")

        submit_button.click(
            fn=gradio_interface,
            inputs=[root_folder, models, output_csv_path, device_map, torch_dtype],
            outputs=[output_html, download_button]
        )

    demo.launch(server_name="0.0.0.0",)