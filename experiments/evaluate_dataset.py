import argparse
import pandas as pd
import torch
import os
import sys

sys.path.append(os.getcwd())
from tqdm import tqdm
from src.pipeline import CreativityPipeline

def evaluate_dataset(metadata_csv, output_file, max_samples=None, num_steps=20):
    try:
        df = pd.read_csv(metadata_csv)
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    if max_samples:
        df = df.iloc[:max_samples]
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = CreativityPipeline(device=device)
    
    processed_images = set()
    write_header = True
    if os.path.exists(output_file):
        try:
            existing_df = pd.read_csv(output_file)
            if 'filename' in existing_df.columns:
                processed_images = set(existing_df['filename'].values)
            write_header = False
        except Exception:
            pass
    
    buffer = []
    FLUSH_EVERY = 10
    
    with open(output_file, 'a' if not write_header else 'w') as f_out:
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            genre = row.get('genre', 'unknown')
            human_fname = row['original_image']
            ai_fname = row['generated_image']
            
            human_path = row['human_path']
            if human_fname not in processed_images and os.path.exists(human_path):
                try:
                    res_human = pipeline.measure(human_path, num_steps=num_steps)
                    if res_human:
                        res_human['label'] = 'human'
                        res_human['genre'] = genre
                        res_human['filename'] = human_fname
                        if 'reconstructed_image' in res_human:
                            del res_human['reconstructed_image']
                        buffer.append(res_human)
                except Exception:
                    continue

            ai_path = row['ai_path']
            if ai_fname not in processed_images and os.path.exists(ai_path):
                try:
                    res_ai = pipeline.measure(ai_path, num_steps=num_steps)
                    if res_ai:
                        res_ai['label'] = 'ai'
                        res_ai['genre'] = genre
                        res_ai['filename'] = ai_fname
                        if 'reconstructed_image' in res_ai:
                            del res_ai['reconstructed_image']
                        buffer.append(res_ai)
                except Exception:
                    continue
            
            if len(buffer) >= FLUSH_EVERY or idx == len(df) - 1:
                if buffer:
                    temp_df = pd.DataFrame(buffer)
                    temp_df.to_csv(f_out, header=write_header, index=False)
                    f_out.flush()
                    write_header = False
                    buffer = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_csv", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="results/wikiart_sd_scores.csv")
    parser.add_argument("--max_samples", type=int, default=3000)
    parser.add_argument("--num_steps", type=int, default=20)
    args = parser.parse_args()
    
    evaluate_dataset(args.metadata_csv, args.output_file, args.max_samples, args.num_steps)
