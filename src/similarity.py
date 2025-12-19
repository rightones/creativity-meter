import numpy as np
import torch
import lpips
from PIL import Image
import torchvision.transforms.functional as F

def cosine_similarity(embed1: np.ndarray, embed2: np.ndarray) -> float:
    if embed1.ndim > 1 or embed2.ndim > 1:
        raise ValueError("Use batch_similarity for multi-dim inputs")
        
    dot = np.dot(embed1, embed2)
    norm1 = np.linalg.norm(embed1)
    norm2 = np.linalg.norm(embed2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
        
    return float(dot / (norm1 * norm2))

def creativity_score(similarity: float) -> float:
    return 1.0 - similarity

def batch_similarity(embeds1: np.ndarray, embeds2: np.ndarray) -> np.ndarray:
    if embeds1.shape != embeds2.shape:
        raise ValueError(f"Shapes mismatch: {embeds1.shape} vs {embeds2.shape}")
        
    norms1 = np.linalg.norm(embeds1, axis=1, keepdims=True)
    norms2 = np.linalg.norm(embeds2, axis=1, keepdims=True)
    
    e1 = embeds1 / (norms1 + 1e-8)
    e2 = embeds2 / (norms2 + 1e-8)
    
    sims = np.sum(e1 * e2, axis=1)
    return sims

def calculate_mse(image1: Image.Image, image2: Image.Image) -> float:
    if image1.size != image2.size:
        image2 = image2.resize(image1.size, resample=Image.BICUBIC)
        
    arr1 = np.array(image1).astype(np.float32)
    arr2 = np.array(image2).astype(np.float32)
    
    mse = np.mean((arr1 - arr2) ** 2)
    return float(mse)

class LPIPSCalculator:
    def __init__(self, device="cuda"):
        self.device = device
        try:
            self.loss_fn = lpips.LPIPS(net='alex').to(device)
            self.loss_fn.eval()
        except Exception as e:
            print(f"Error loading LPIPS: {e}")
            raise e
            
    def calculate(self, image1: Image.Image, image2: Image.Image) -> float:
        if image1.size != image2.size:
             image2 = image2.resize(image1.size, resample=Image.BICUBIC)
             
        t1 = F.to_tensor(image1).to(self.device) * 2 - 1 
        t2 = F.to_tensor(image2).to(self.device) * 2 - 1
        
        t1 = t1.unsqueeze(0)
        t2 = t2.unsqueeze(0)
        
        with torch.no_grad():
            dist = self.loss_fn(t1, t2)
            
        return float(dist.item())

def calculate_composite_score(similarity: float, mse: float, lpips_val: float) -> float:
    s1 = 1.0 - similarity
    s2 = lpips_val
    s3 = np.log10(1 + mse) / 5.0
    
    return (s1 + s2 + s3) / 3.0
