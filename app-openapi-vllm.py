import os
import base64
import io
import pandas as pd
from PIL import Image
import requests
import gradio as gr
from tqdm import tqdm
import gc

# Clean GPU memory utility
def clean_memory():
    """Utility to clean GPU memory"""
    gc.collect()

# Define constants for models
available_models = [
    "llava-hf/llava-1.5-7b-hf",
    "llava-hf/llava-v1.6-mistral-7b-hf",
    "Qwen/Qwen2-VL-7B-Instruct",
]

# Prompt for person detection
person_detection_prompt = (
    "Analyze the person in the provided image in detail. Describe their posture (e.g., standing, sitting), clothing (color, type), and visible accessories. "
    "Include details like their position in the image (e.g., center, left) and items they are carrying (e.g., bag, phone). Do not refuse to analyze the image."
)

def encode_image(image):
    """Encode an image to a base64 string."""
    img_buffer = io.BytesIO()
    image.save(img_buffer, format="PNG")
    return base64.b64encode(img_buffer.getvalue()).decode("utf-8")

def generate_output(prompt, image, model_name, vllm_server_address):
    """Generate output using the vLLM API."""
    try:
        # Encode the image to base64
        img_str = encode_image(image)
        headers = {"Content-Type": "application/json"}
        # Prepare the payload for the API
        payload = {
            "model": model_name,  # Specify the model
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_str}"  # Correct format
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.9
        }

        # Make the API call
        response = requests.post(f"{vllm_server_address}/v1/chat/completions", json=payload, headers=headers)
        response.raise_for_status()

        # Extract the response text
        response_text = response.json()["choices"][0]["message"]["content"]
        
        # Check for refusal responses
        if "unable to identify" in response_text.lower() or "sorry" in response_text.lower():
            return "ERROR: Model refused to analyze the image."
        return response_text

    except Exception as e:
        return f"Error during generation: {str(e)}"

def test_vllm_connection(vllm_server_address):
    """Test the connection to the vLLM server and list available models."""
    try:
        # Test connection by hitting the /health endpoint
        health_response = requests.get(f"{vllm_server_address}/health")
        health_response.raise_for_status()
        connection_status = "Connection successful!\n\n"

        # Fetch the list of models from the /v1/models endpoint
        models_response = requests.get(f"{vllm_server_address}/v1/models")
        models_response.raise_for_status()

        # Extract the list of models from the response
        models = models_response.json().get("data", [])
        model_list = "\n".join([model["id"] for model in models])

        return f"{connection_status}Available models:\n{model_list}"
    except Exception as e:
        return f"Connection failed: {str(e)}"

def get_supported_models(vllm_server_address):
    """Query the vLLM server to get the list of supported models."""
    try:
        # Fetch the list of models from the /v1/models endpoint
        models_response = requests.get(f"{vllm_server_address}/v1/models")
        models_response.raise_for_status()

        # Extract the list of models from the response
        models = models_response.json().get("data", [])
        return [model["id"] for model in models]
    except Exception as e:
        print(f"Error fetching models from vLLM server {vllm_server_address}: {e}")
        return []

def process_images_from_root_folder(root_folder, model_name, prompt, output_csv_path, vllm_server_address, progress=None):
    """Process images from the root folder and generate predictions."""
    # Initialize df to an empty DataFrame
    df = pd.DataFrame()

    data = []
    images = []

    # Walk through the root folder
    for root, dirs, files in os.walk(root_folder):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, filename)
                try:
                    img = Image.open(image_path).convert("RGB")
                    # Resize image to create a thumbnail
                    img.thumbnail((100, 100))  # Resize to 100x100 pixels
                    images.append({
                        "filename": filename,
                        "image": img,
                        "folder_path": root
                    })
                except Exception as e:
                    print(f"Error opening image {filename}: {e}")

    for idx, img in enumerate(tqdm(images, desc="Processing images")):
        print(f"\nProcessing: {os.path.join(img['folder_path'], img['filename'])}")

        # Initialize row data
        row_data = {
            "Image_Name": img['filename'],
            "Description": "",
            "Thumbnail": f'<img src="data:image/png;base64,{encode_image(img["image"])}" width="100" height="100">'
        }

        try:
            response_text = generate_output(prompt, img['image'], model_name, vllm_server_address)
            print(f"Model {model_name} response: {response_text}")

            # Store the description
            row_data["Description"] = response_text

        except Exception as e:
            print(f"Error processing with model {model_name}: {e}")
            row_data["Description"] = "ERROR"
        
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

def gradio_interface(root_folder, output_csv_path, selected_model, progress=gr.Progress()):
    """Gradio interface for processing images."""
    # Get the vLLM server address corresponding to the selected model
    vllm_server_address = model_to_server_mapping.get(selected_model, "")
    if not vllm_server_address:
        return "Error: No server found for the selected model.", ""

    print(f"Using model: {selected_model} from server: {vllm_server_address}")

    df, csv_path = process_images_from_root_folder(root_folder, selected_model, person_detection_prompt, output_csv_path, vllm_server_address, progress)

    # Convert the DataFrame to HTML and include images
    html = df.to_html(classes='table table-striped', index=False, escape=False)

    return html, csv_path

def load_first_5_images(root_folder):
    """Load the first 5 images from the root folder and return them as a list of file paths."""
    images = []
    for root, _, files in os.walk(root_folder):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, filename)
                images.append(image_path)
                if len(images) >= 5:
                    break
        if len(images) >= 5:
            break
    return images

if __name__ == "__main__":
    # List of vLLM server addresses
    vllm_server_addresses = [
        "http://127.0.0.1:8000",
    ]

    # Fetch available models from all vLLM servers
    model_to_server_mapping = {}
    for server_address in vllm_server_addresses:
        models = get_supported_models(server_address)
        for model in models:
            model_to_server_mapping[model] = server_address

    # Extract the list of available models
    available_models = list(model_to_server_mapping.keys())

    # Launch the Gradio UI
    with gr.Blocks() as demo:
        gr.Markdown("# vLLM - Person Detection")
        gr.Markdown(f"**Prompt:** {person_detection_prompt}")

        # Test connection button
        gr.Markdown("## VLM Models")
        test_connection_button = gr.Button("Test Connection")

        with gr.Row():
            model_dropdown = gr.Dropdown(label="Available Models", choices=available_models,
                                      value=available_models[0] if available_models else "")
            connection_status_box = gr.Textbox(label="Connection Status", interactive=False)

        gr.Markdown("## Image Processing")
        submit_button = gr.Button("Process Images")
        with gr.Row():
            root_folder = gr.Textbox(label="Images Folder", value=os.path.expanduser("~/app/images/"))
            show_images_button = gr.Button("Show Images")
        image_grid = gr.Gallery(label="First 5 Images", columns=5)

        output_csv_path = gr.Textbox(label="Output CSV Path", value="person_detection.csv")

        with gr.Tabs():
            with gr.Tab("Results"):
                output_html = gr.HTML()

        download_button = gr.File(label="Download CSV")

        # Test connection button click event
        test_connection_button.click(
            fn=lambda model: test_vllm_connection(model_to_server_mapping.get(model, "")),
            inputs=model_dropdown,
            outputs=connection_status_box
        )

        # Submit button click event
        submit_button.click(
            fn=gradio_interface,
            inputs=[root_folder, output_csv_path, model_dropdown],
            outputs=[output_html, download_button]
        )

        # Show images button click event
        show_images_button.click(
            fn=load_first_5_images,
            inputs=root_folder,
            outputs=image_grid
        )

    demo.launch(server_name="0.0.0.0",)