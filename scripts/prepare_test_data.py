import os
import torch
import requests
from diffusers import StableDiffusionPipeline
from PIL import Image
from io import BytesIO
from tqdm import tqdm

def setup_dirs(data_dir):
    os.makedirs(os.path.join(data_dir, "ai"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "human"), exist_ok=True)

def generate_ai_images(data_dir, num_images=10, device="cuda"):
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to(device)
    
    prompts = [
        "a photo of a cute cat sitting on a sofa",
        "a beautiful landscape of mountains and lake",
        "a futuristic city with flying cars",
        "a portrait of a smiling woman",
        "a bowl of fresh fruit on a table",
        "a red sports car driving on a highway",
        "a cozy wooden cabin in the forest",
        "a robot playing chess",
        "a delicious pizza with pepperoni",
        "a sunset over the ocean"
    ]
    
    for i in tqdm(range(num_images)):
        prompt = prompts[i % len(prompts)]
        image = pipe(prompt).images[0]
        save_path = os.path.join(data_dir, "ai", f"ai_gen_{i:03d}.png")
        image.save(save_path)

def download_human_images(data_dir, num_images=10):
    urls = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/640px-Cat_November_2010-1a.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c8/Altja_j%C3%B5gi_Lahemaal.jpg/640px-Altja_j%C3%B5gi_Lahemaal.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e6/Paris_Night.jpg/640px-Paris_Night.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b2/Vincent_van_Gogh_-_Self-Portrait_-_Google_Art_Project.jpg/640px-Vincent_van_Gogh_-_Self-Portrait_-_Google_Art_Project.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6d/Good_Food_Display_-_NCI_Visuals_Online.jpg/640px-Good_Food_Display_-_NCI_Visuals_Online.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/2019_Toyota_Supra_GR_3.0_Pro_front.jpg/640px-2019_Toyota_Supra_GR_3.0_Pro_front.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Log_Cabin_-_panoramio.jpg/640px-Log_Cabin_-_panoramio.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f2/Chess_set_-_Staunton_No._6_-_01.jpg/640px-Chess_set_-_Staunton_No._6_-_01.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a3/Eq_it-na_pizza-margherita_sep2005_sml.jpg/640px-Eq_it-na_pizza-margherita_sep2005_sml.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Sunset_in_the_fields_by_Sendi.jpg/640px-Sunset_in_the_fields_by_Sendi.jpg"
    ]
    
    headers = {'User-Agent': 'Mozilla/5.0'}
    count = 0
    for i, url in enumerate(tqdm(urls)):
        if count >= num_images:
            break
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                img = Image.open(BytesIO(resp.content)).convert("RGB")
                save_path = os.path.join(data_dir, "human", f"human_real_{i:03d}.jpg")
                img.save(save_path)
                count += 1
        except Exception:
            continue

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    args = parser.parse_args()
    
    setup_dirs(args.data_dir)
    generate_ai_images(args.data_dir)
    download_human_images(args.data_dir)
