import argparse
import os
import pandas as pd
import shutil
import tarfile
from huggingface_hub import hf_hub_download
from tqdm import tqdm

def download_wikiart_sd(output_dir, max_samples=None):
    ai_repo_id = "Dant33/Wikiart_with_StableDiffusion"
    ai_tar_path = hf_hub_download(repo_id=ai_repo_id, filename="dataset.tar.gz", repo_type="dataset", cache_dir=os.path.join(output_dir, "cache"))
    
    human_repo_id = "Dant33/WikiArt-81K-BLIP_2-768x768"
    human_tar_path = hf_hub_download(repo_id=human_repo_id, filename="dataset.tar.gz", repo_type="dataset", cache_dir=os.path.join(output_dir, "cache"))
    
    ai_tar = tarfile.open(ai_tar_path, "r:gz")
    human_tar = tarfile.open(human_tar_path, "r:gz")
    
    metadata_member = None
    for member in ai_tar:
        if member.name.endswith("data.csv"):
            metadata_member = member
            break
            
    if not metadata_member:
        raise FileNotFoundError("data.csv not found in AI archive")
    
    ai_tar.extract(metadata_member, path=output_dir)
    metadata_path = os.path.join(output_dir, metadata_member.name)
    
    df = pd.read_csv(metadata_path)
    
    if 'original_image' not in df.columns:
        def get_original(gen_path):
            return gen_path.replace("_generated", "")

        df['original_image'] = df['generated_image'].apply(get_original)
        df['genre'] = df['generated_image'].apply(lambda x: x.split('/')[1] if '/' in x else 'unknown')

    if max_samples:
        df = df.iloc[:max_samples]

    final_images_dir = os.path.join(output_dir, "images")
    human_out_dir = os.path.join(final_images_dir, "human")
    ai_out_dir = os.path.join(final_images_dir, "ai")
    os.makedirs(human_out_dir, exist_ok=True)
    os.makedirs(ai_out_dir, exist_ok=True)
    
    meta_map = {}
    for idx, row in df.iterrows():
        gen_base = os.path.basename(row['generated_image'])
        orig_base = os.path.basename(row['original_image'])
        meta_map[gen_base] = {
            'row': row,
            'orig_base': orig_base,
            'gen_base': gen_base
        }
    
    extracted_ai = set()
    extracted_human = set()
    
    def scan_and_extract(tar, out_dir, target_basenames, extracted_set, label):
        for member in tqdm(tar, desc=f"Scanning {label}"):
            if member.isfile():
                base = os.path.basename(member.name)
                if base in target_basenames:
                    target_path = os.path.join(out_dir, base)
                    if not os.path.exists(target_path):
                        f_in = tar.extractfile(member)
                        with open(target_path, "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    extracted_set.add(base)
                    
    ai_targets = set(meta_map.keys())
    ai_tar.close()
    ai_tar = tarfile.open(ai_tar_path, "r:gz")
    
    scan_and_extract(ai_tar, ai_out_dir, ai_targets, extracted_ai, "AI")
    ai_tar.close()
    
    human_targets = set(v['orig_base'] for v in meta_map.values())
    human_tar.close()
    human_tar = tarfile.open(human_tar_path, "r:gz")
    scan_and_extract(human_tar, human_out_dir, human_targets, extracted_human, "Human")
    human_tar.close()
    
    results = []
    for gen_base, info in meta_map.items():
         orig_base = info['orig_base']
         if gen_base in extracted_ai and orig_base in extracted_human:
             row = info['row']
             human_path = os.path.join(human_out_dir, orig_base)
             ai_path = os.path.join(ai_out_dir, gen_base)
             
             results.append({
                'original_image': orig_base,
                'generated_image': gen_base,
                'human_path': human_path,
                'ai_path': ai_path,
                'genre': row.get('genre'),
                'artist': row.get('artist'),
                'style': row.get('style'),
                'title': row.get('title')
            })
            
    results_df = pd.DataFrame(results)
    new_csv_path = os.path.join(output_dir, "dataset_map.csv")
    results_df.to_csv(new_csv_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/wikiart_sd")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    
    download_wikiart_sd(args.output_dir, args.max_samples)
