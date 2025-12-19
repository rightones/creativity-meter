import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def generate_report(results_csv, output_report_path="report.md"):
    print(f"Loading results from {results_csv}...")
    df = pd.read_csv(results_csv)
    
    # Calculate Creativity Score (1 - Similarity) if not present
    # Assuming pipeline returns 'similarity' (cosine similarity)
    # The higher the similarity, the lower the creativity (reconstruction error is low).
    # Creativity = 1 - Similarity (roughly).
    
    if 'similarity' in df.columns:
        df['creativity_score'] = 1.0 - df['similarity']
        
    # Group by Label (Human vs AI)
    summary = df.groupby('label')['creativity_score'].describe()
    print("Summary Statistics:")
    print(summary)
    
    # T-test
    from scipy.stats import ttest_ind
    human_scores = df[df['label'] == 'human']['creativity_score']
    ai_scores = df[df['label'] == 'ai']['creativity_score']
    
    t_stat, p_val = ttest_ind(human_scores, ai_scores, equal_var=False)
    print(f"T-test: t={t_stat:.2f}, p={p_val:.4f}")
    
    # Generate Plots
    report_dir = os.path.dirname(output_report_path)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    else:
        report_dir = "."
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='creativity_score', hue='label', kde=True, bins=30)
    plt.title('Distribution of Creativity Scores (Human vs AI)')
    plt.savefig(os.path.join(report_dir, 'score_distribution.png'))
    plt.close()
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='genre', y='creativity_score', hue='label')
    plt.xticks(rotation=45)
    plt.title('Creativity Scores by Genre')
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, 'score_by_genre.png'))
    plt.close()
    
    # Write Markdown Report
    with open(output_report_path, "w") as f:
        f.write("# Creativity Hypothesis Analysis Report\n\n")
        f.write("## Summary Statistics\n")
        f.write(summary.to_markdown())
        f.write(f"\n\n**Statistical Test (T-test):** t={t_stat:.2f}, p={p_val:.4f}\n")
        if p_val < 0.05:
            f.write("Result is statistically significant.\n")
        else:
            f.write("Result is NOT statistically significant.\n")
            
        f.write("\n## visualizations\n")
        f.write("![Score Distribution](score_distribution.png)\n")
        f.write("![Score by Genre](score_by_genre.png)\n")
        
    print(f"Report generated at {output_report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_csv", type=str, required=True)
    parser.add_argument("--output_report", type=str, default="report.md")
    args = parser.parse_args()
    
    generate_report(args.results_csv, args.output_report)
