import os
import glob
import pandas as pd
from tqdm import tqdm
from src.pipeline import CreativityPipeline
import argparse
import random

def load_genimage_subset(data_dir: str, n_samples: int = 100) -> dict:
    ai_dir = os.path.join(data_dir, "ai")
    human_dir = os.path.join(data_dir, "human")
    
    if not os.path.exists(ai_dir) or not os.path.exists(human_dir):
        all_files = glob.glob(os.path.join(data_dir, "**", "*.[jJ][pP][gG]"), recursive=True)
        if not all_files:
             all_files = glob.glob(os.path.join(data_dir, "**", "*.[pP][nN][gG]"), recursive=True)
        
        random.shuffle(all_files)
        mid = len(all_files) // 2
        return {
            "ai": all_files[:min(n_samples, mid)],
            "human": all_files[min(n_samples, mid):min(n_samples*2, len(all_files))]
        }

    exts = ['*.jpg', '*.jpeg', '*.png', '*.webp']
    ai_files = []
    human_files = []
    
    for ext in exts:
        ai_files.extend(glob.glob(os.path.join(ai_dir, ext)))
        ai_files.extend(glob.glob(os.path.join(ai_dir, ext.upper())))
        human_files.extend(glob.glob(os.path.join(human_dir, ext)))
        human_files.extend(glob.glob(os.path.join(human_dir, ext.upper())))
    
    random.shuffle(ai_files)
    random.shuffle(human_files)
    
    return {
        "ai": ai_files[:n_samples],
        "human": human_files[:n_samples]
    }

def run_experiment(data_dir: str, output_dir: str, n_samples: int = 100, device: str = "cuda"):
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = load_genimage_subset(data_dir, n_samples)
    
    if len(dataset['ai']) == 0 and len(dataset['human']) == 0:
        return

    pipeline = CreativityPipeline(device=device)
    results = []
    
    for path in tqdm(dataset['ai'], desc="AI"):
        res = pipeline.measure(path)
        if res:
            res['label'] = 'ai'
            if 'reconstructed_image' in res:
                del res['reconstructed_image']
            results.append(res)
            
    for path in tqdm(dataset['human'], desc="Human"):
        res = pipeline.measure(path)
        if res:
            res['label'] = 'human'
            if 'reconstructed_image' in res:
                del res['reconstructed_image']
            results.append(res)
            
    df = pd.DataFrame(results)
    save_path = os.path.join(output_dir, "experiment_results.csv")
    df.to_csv(save_path, index=False)
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    run_experiment(args.data_dir, args.output_dir, args.n_samples, args.device)
