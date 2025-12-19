import argparse
import os
import glob
from src.pipeline import CreativityPipeline
import pandas as pd
import torch

def main():
    parser = argparse.ArgumentParser(description="Creativity Measurement Tool")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    measure_parser = subparsers.add_parser("measure", help="Measure creativity of image(s)")
    measure_parser.add_argument("--image", type=str, help="Path to single image")
    measure_parser.add_argument("--dir", type=str, help="Directory of images")
    measure_parser.add_argument("--output", type=str, default="results.csv", help="Output CSV file")
    measure_parser.add_argument("--save_recon", type=str, help="Directory to save reconstructed images")
    measure_parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    
    args = parser.parse_args()
    
    if args.command == "measure":
        pipeline = CreativityPipeline(device=args.device)
        
        if args.image:
            print(f"Measuring single image: {args.image}")
            res = pipeline.measure(args.image)
            if res:
                print("-" * 30)
                print(f"Original: {args.image}")
                print(f"Similarity: {res['similarity']:.4f}")
                print(f"Creativity Score: {res['creativity_score']:.4f}")
                print("-" * 30)
                
                if args.save_recon:
                    os.makedirs(args.save_recon, exist_ok=True)
                    fname = os.path.basename(args.image)
                    save_path = os.path.join(args.save_recon, f"recon_{fname}")
                    res['reconstructed_image'].save(save_path)
                    print(f"Reconstructed image saved to {save_path}")
                    
        elif args.dir:
            exts = ['*.jpg', '*.jpeg', '*.png', '*.webp']
            files = []
            for ext in exts:
                files.extend(glob.glob(os.path.join(args.dir, ext)))
                files.extend(glob.glob(os.path.join(args.dir, ext.upper())))
            
            print(f"Found {len(files)} images in {args.dir}")
            if files:
                df = pipeline.measure_batch(files, save_recon_dir=args.save_recon)
                print(f"Saving results to {args.output}")
                df.to_csv(args.output, index=False)
                
                print("Summary:")
                print(df[['similarity', 'creativity_score']].describe())
        else:
            print("Please provide --image or --dir")
            
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
