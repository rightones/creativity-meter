import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import os

def comprehensive_analysis(results_csv, output_dir="results/comprehensive_analysis"):
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(results_csv)
    
    if 'creativity_score' not in df.columns and 'similarity' in df.columns:
        df['creativity_score'] = 1.0 - df['similarity']
    
    print(f"Total samples: {len(df)}")
    print(f"AI samples: {len(df[df['label'] == 'ai'])}")
    print(f"Human samples: {len(df[df['label'] == 'human'])}")
    
    print("\n1. Statistics")
    for label in ['ai', 'human']:
        subset = df[df['label'] == label]
        print(f"\n[{label.upper()}]")
        for col in ['similarity', 'creativity_score', 'mse', 'lpips', 'composite_score']:
            if col in subset.columns:
                print(f"  {col}: mean={subset[col].mean():.4f}, std={subset[col].std():.4f}, median={subset[col].median():.4f}")
    
    print("\n2. Statistical Tests")
    ai_data = df[df['label'] == 'ai']
    human_data = df[df['label'] == 'human']
    
    test_results = {}
    for col in ['similarity', 'creativity_score', 'mse', 'lpips', 'composite_score']:
        if col in df.columns:
            t_stat, p_val = stats.ttest_ind(ai_data[col], human_data[col], equal_var=False)
            u_stat, u_pval = stats.mannwhitneyu(ai_data[col], human_data[col], alternative='two-sided')
            test_results[col] = {
                't_stat': t_stat, 't_pval': p_val,
                'u_stat': u_stat, 'u_pval': u_pval
            }
            print(f"\n[{col}]")
            print(f"  T-test: t={t_stat:.4f}, p={p_val:.6f}")
            print(f"  Mann-Whitney U: U={u_stat:.2f}, p={u_pval:.6f}")
    
    if 'genre' in df.columns:
        print("\n3. Genre-wise Analysis")
        genre_stats = df.groupby(['genre', 'label']).agg({
            'creativity_score': ['mean', 'std', 'count']
        }).round(4)
        print(genre_stats)
        
        plt.figure(figsize=(14, 8))
        genre_means = df.groupby(['genre', 'label'])['creativity_score'].mean().unstack()
        genre_means.plot(kind='bar', ax=plt.gca())
        plt.title('Creativity Score by Genre')
        plt.xlabel('Genre')
        plt.ylabel('Mean Creativity Score')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'genre_comparison.png'), dpi=150)
        plt.close()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, col in enumerate(['similarity', 'creativity_score', 'mse', 'lpips']):
        if col in df.columns:
            ax = axes[idx // 2, idx % 2]
            for label, color in [('ai', 'red'), ('human', 'blue')]:
                subset = df[df['label'] == label][col]
                ax.hist(subset, bins=30, alpha=0.5, label=label, color=color, density=True)
            ax.set_title(f'Distribution of {col}')
            ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metric_distributions.png'), dpi=150)
    plt.close()
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for idx, col in enumerate(['similarity', 'creativity_score', 'mse', 'lpips']):
        if col in df.columns:
            sns.boxplot(data=df, x='label', y=col, ax=axes[idx], palette={'ai': 'red', 'human': 'blue'})
            axes[idx].set_title(col)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metric_boxplots.png'), dpi=150)
    plt.close()
    
    numeric_cols = ['similarity', 'creativity_score', 'mse', 'lpips', 'composite_score']
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    plt.figure(figsize=(10, 8))
    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.3f')
    plt.title('Correlation')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=150)
    plt.close()
    
    def cohens_d(group1, group2):
        n1, n2 = len(group1), len(group2)
        var1, var2 = group1.var(), group2.var()
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        return (group1.mean() - group2.mean()) / pooled_std
    
    print("\n4. Effect Size")
    for col in ['similarity', 'creativity_score', 'mse', 'lpips']:
        if col in df.columns:
            d = cohens_d(ai_data[col], human_data[col])
            print(f"  {col}: d={d:.4f}")
    
    report_path = os.path.join(output_dir, 'comprehensive_report.md')
    with open(report_path, 'w') as f:
        f.write("# Analysis Report\n\n")
        f.write(f"Total Samples: {len(df)} (AI: {len(ai_data)}, Human: {len(human_data)})\n\n")
        
        ai_mean = ai_data['creativity_score'].mean()
        human_mean = human_data['creativity_score'].mean()
        
        f.write(f"### Creativity Score\n")
        f.write(f"- AI Mean: {ai_mean:.4f}\n")
        f.write(f"- Human Mean: {human_mean:.4f}\n\n")
        
        if test_results.get('creativity_score'):
            tr = test_results['creativity_score']
            f.write(f"- T-test: p={tr['t_pval']:.6f}\n")
            f.write(f"- Mann-Whitney U: p={tr['u_pval']:.6f}\n\n")
        
        f.write("## Visualizations\n\n")
        f.write("![Metric Distributions](metric_distributions.png)\n")
        f.write("![Metric Boxplots](metric_boxplots.png)\n")
        f.write("![Correlation Heatmap](correlation_heatmap.png)\n")
        if 'genre' in df.columns:
            f.write("![Genre Comparison](genre_comparison.png)\n")

if __name__ == "__main__":
    comprehensive_analysis("results/wikiart_sd_full_scores.csv")
