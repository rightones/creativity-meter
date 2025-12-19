import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import argparse
import os
import numpy as np

def load_results(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)

def statistical_analysis(df: pd.DataFrame) -> dict:
    metrics = ['creativity_score', 'mse', 'lpips', 'composite_score']
    results = {}
    
    for metric in metrics:
        if metric not in df.columns:
            continue
            
        ai_scores = df[df['label'] == 'ai'][metric]
        human_scores = df[df['label'] == 'human'][metric]
        
        # t-test
        t_stat, p_val = stats.ttest_ind(ai_scores, human_scores, equal_var=False)
        
        results[f"{metric}_ai_mean"] = np.mean(ai_scores)
        results[f"{metric}_human_mean"] = np.mean(human_scores)
        results[f"{metric}_p_value"] = p_val
    
    return results

def visualize_results(df: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = ['creativity_score', 'mse', 'lpips', 'composite_score']
    titles = ['Creativity Score (1 - CosSim)', 'MSE (Pixel Error)', 'LPIPS (Perceptual Error)', 'Composite Score']
    
    for metric, title in zip(metrics, titles):
        if metric not in df.columns:
            continue
            
        ai_scores = df[df['label'] == 'ai'][metric]
        human_scores = df[df['label'] == 'human'][metric]
        
        # Boxplot
        plt.figure(figsize=(8, 6))
        plt.boxplot([ai_scores, human_scores], labels=['AI', 'Human'])
        plt.title(f'Comparison: {title}')
        plt.ylabel('Score')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f'boxplot_{metric}.png'))
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to results csv")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory for plots")
    
    args = parser.parse_args()
    
    df = load_results(args.csv)
    stats_res = statistical_analysis(df)
    
    print("-" * 30)
    print("Statistical Analysis Results:")
    for k, v in stats_res.items():
        print(f"{k}: {v:.4f}")
    print("-" * 30)
    
    visualize_results(df, args.output_dir)
    print(f"Plots saved to {args.output_dir}")

if __name__ == "__main__":
    main()
