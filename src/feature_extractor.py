import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import numpy as np

class DINOv2Extractor:
    def __init__(self, model_name="facebook/dinov2-base", device="cuda"):
        self.device = device
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(device)
        except Exception as e:
            raise e
            
        self.model.eval()
    
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs['pixel_values'].to(self.device)
    
    def extract(self, image: Image.Image) -> np.ndarray:
        pixel_values = self.preprocess(image)
        with torch.no_grad():
            outputs = self.model(pixel_values)
            embedding = outputs.last_hidden_state[:, 0, :]
            
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy()[0]
    
    def extract_batch(self, images: list[Image.Image]) -> np.ndarray:
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.device)
        with torch.no_grad():
            outputs = self.model(pixel_values)
            embeddings = outputs.last_hidden_state[:, 0, :]
            
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.cpu().numpy()
