import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
import numpy as np
from tqdm import tqdm

class DDIMInversion:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device="cuda", dtype=torch.float16):
        self.device = device
        self.dtype = dtype
        
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=dtype,
            use_safetensors=True,
        ).to(device)
        
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        
        self.vae = self.pipe.vae
        self.unet = self.pipe.unet
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        self.scheduler = self.pipe.scheduler

    def encode_image(self, image: Image.Image) -> torch.Tensor:
        width, height = image.size
        min_dim = min(width, height)
        left = (width - min_dim) / 2
        top = (height - min_dim) / 2
        right = (width + min_dim) / 2
        bottom = (height + min_dim) / 2
        image = image.crop((left, top, right, bottom))
        
        image = image.resize((512, 512), resample=Image.BICUBIC)
        img_np = np.array(image).astype(np.float32) / 127.5 - 1.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=self.dtype)
        
        with torch.no_grad():
            latent = self.vae.config.scaling_factor * self.vae.encode(img_tensor).latent_dist.sample()
        return latent

    def decode_latent(self, latent: torch.Tensor) -> Image.Image:
        with torch.no_grad():
            image = self.vae.decode(latent / self.vae.config.scaling_factor).sample
            
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).astype(np.uint8)
        return Image.fromarray(image[0])
        
    def invert(self, image: Image.Image, num_steps: int = 50) -> torch.Tensor:
        latents = self.encode_image(image)
        
        self.scheduler.set_timesteps(num_steps)
        
        prompt = ""
        text_input = self.tokenizer(
            prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        timesteps = self.scheduler.timesteps.tolist()
        reversed_timesteps = timesteps[::-1]
        
        with torch.no_grad():
            for i, t in enumerate(reversed_timesteps):
                noise_pred = self.unet(latents, t, encoder_hidden_states=text_embeddings).sample
                
                if i < len(reversed_timesteps) - 1:
                    next_t = reversed_timesteps[i + 1]
                else:
                    next_t = 999 
                
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_next = self.scheduler.alphas_cumprod[next_t]
                
                beta_prod_t = 1 - alpha_prod_t
                pred_original_sample = (latents - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5
                
                pred_epsilon = noise_pred
                
                beta_prod_next = 1 - alpha_prod_next
                latents = alpha_prod_next ** 0.5 * pred_original_sample + beta_prod_next ** 0.5 * pred_epsilon
                
        return latents

    def reconstruct(self, noise: torch.Tensor, num_steps: int = 50) -> Image.Image:
        return self.pipe(
             prompt="",
             num_inference_steps=num_steps,
             guidance_scale=0.0,
             latents=noise,
             output_type="pil"
        ).images[0]

    def invert_and_reconstruct(self, image: Image.Image, num_steps: int = 50) -> tuple[Image.Image, dict]:
        noise = self.invert(image, num_steps=num_steps)
        recon = self.reconstruct(noise, num_steps=num_steps)
        return recon, {"noise": noise}
