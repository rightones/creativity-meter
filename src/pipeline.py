import torch
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm

from src.inversion import DDIMInversion
from src.feature_extractor import DINOv2Extractor
from src.similarity import cosine_similarity, creativity_score, calculate_mse, LPIPSCalculator, calculate_composite_score

class CreativityPipeline:
    def __init__(self, device="cuda"):
        self.device = device
        self.inverter = DDIMInversion(device=device)
        self.extractor = DINOv2Extractor(device=device)
        self.lpips_calc = LPIPSCalculator(device=device)
    
    def measure(self, image_path: str, num_steps: int = 50) -> dict:
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
            
        recon_image, _ = self.inverter.invert_and_reconstruct(image, num_steps=num_steps)
        
        emb_orig = self.extractor.extract(image)
        emb_recon = self.extractor.extract(recon_image)
        
        sim = cosine_similarity(emb_orig, emb_recon)
        score = creativity_score(sim)
        mse = calculate_mse(image, recon_image)
        lpips_val = self.lpips_calc.calculate(image, recon_image)
        composite = calculate_composite_score(sim, mse, lpips_val)
        
        return {
            "original_path": image_path,
            "similarity": sim,
            "creativity_score": score,
            "mse": mse,
            "lpips": lpips_val,
            "composite_score": composite,
            "reconstructed_image": recon_image
        }
    
    def measure_batch(self, image_paths: list[str], save_recon_dir: str = None) -> pd.DataFrame:
        results = []
        if save_recon_dir:
            os.makedirs(save_recon_dir, exist_ok=True)
            
        for path in tqdm(image_paths):
            res = self.measure(path)
            if res:
                if save_recon_dir:
                    fname = os.path.basename(path)
                    res["reconstructed_image"].save(os.path.join(save_recon_dir, f"recon_{fname}"))
                
                item = res.copy()
                del item["reconstructed_image"]
                results.append(item)
                
        return pd.DataFrame(results)
