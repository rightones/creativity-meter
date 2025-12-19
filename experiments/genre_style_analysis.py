"""
Genre and Style Breakdown Analysis
Analyzes creativity scores by artist and inferred art movement/style.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import os
import re

# Mapping of artists to art movements/styles
ARTIST_STYLE_MAP = {
    # Impressionism
    'claude-monet': 'Impressionism',
    'pierre-auguste-renoir': 'Impressionism',
    'camille-pissarro': 'Impressionism',
    'edgar-degas': 'Impressionism',
    'berthe-morisot': 'Impressionism',
    'alfred-sisley': 'Impressionism',
    'frederic-bazille': 'Impressionism',
    'gustave-caillebotte': 'Impressionism',
    'mary-cassatt': 'Impressionism',
    'childe-hassam': 'Impressionism',
    'willard-metcalf': 'Impressionism',
    'maurice-prendergast': 'Impressionism',
    
    # Post-Impressionism
    'vincent-van-gogh': 'Post-Impressionism',
    'paul-gauguin': 'Post-Impressionism',
    'paul-cezanne': 'Post-Impressionism',
    'henri-de-toulouse-lautrec': 'Post-Impressionism',
    'georges-seurat': 'Post-Impressionism',
    'henri-rousseau': 'Post-Impressionism',
    'odilon-redon': 'Post-Impressionism',
    'henri-edmond-cross': 'Post-Impressionism',
    
    # Expressionism
    'edvard-munch': 'Expressionism',
    'egon-schiele': 'Expressionism',
    'ernst-ludwig-kirchner': 'Expressionism',
    'wassily-kandinsky': 'Abstract/Expressionism',
    'franz-marc': 'Expressionism',
    'chaim-soutine': 'Expressionism',
    
    # Baroque
    'rembrandt': 'Baroque',
    'peter-paul-rubens': 'Baroque',
    'anthony-van-dyck': 'Baroque',
    'jacob-jordaens': 'Baroque',
    'johannes-vermeer': 'Baroque',
    'caravaggio': 'Baroque',
    'diego-velazquez': 'Baroque',
    
    # Renaissance
    'leonardo-da-vinci': 'Renaissance',
    'michelangelo': 'Renaissance',
    'raphael': 'Renaissance',
    'sandro-botticelli': 'Renaissance',
    'titian': 'Renaissance',
    'tintoretto': 'Renaissance',
    'paolo-veronese': 'Renaissance',
    'albrecht-durer': 'Renaissance',
    'andrea-del-verrocchio': 'Renaissance',
    'fra-angelico': 'Renaissance',
    'filippo-lippi': 'Renaissance',
    'piero-della-francesca': 'Renaissance',
    'giovanni-bellini': 'Renaissance',
    
    # Romanticism
    'caspar-david-friedrich': 'Romanticism',
    'eugene-delacroix': 'Romanticism',
    'francisco-goya': 'Romanticism',
    'j-m-w-turner': 'Romanticism',
    'john-constable': 'Romanticism',
    'ivan-aivazovsky': 'Romanticism',
    'theodore-gericault': 'Romanticism',
    
    # Realism
    'gustave-courbet': 'Realism',
    'jean-francois-millet': 'Realism',
    'honore-daumier': 'Realism',
    'ilya-repin': 'Realism',
    'ivan-shishkin': 'Realism',
    'isaac-levitan': 'Realism',
    'ivan-kramskoy': 'Realism',
    
    # Cubism/Modernism
    'pablo-picasso': 'Cubism/Modernism',
    'georges-braque': 'Cubism/Modernism',
    'juan-gris': 'Cubism/Modernism',
    'fernand-leger': 'Cubism/Modernism',
    
    # Surrealism
    'salvador-dali': 'Surrealism',
    'rene-magritte': 'Surrealism',
    'max-ernst': 'Surrealism',
    'joan-miro': 'Surrealism',
    'marc-chagall': 'Surrealism',
    
    # Abstract
    'piet-mondrian': 'Abstract',
    'kazimir-malevich': 'Abstract',
    'jackson-pollock': 'Abstract Expressionism',
    'mark-rothko': 'Abstract Expressionism',
    'willem-de-kooning': 'Abstract Expressionism',
    
    # Fauvism
    'henri-matisse': 'Fauvism',
    'andre-derain': 'Fauvism',
    'maurice-de-vlaminck': 'Fauvism',
    
    # Symbolism
    'gustave-moreau': 'Symbolism',
    'gustav-klimt': 'Symbolism/Art Nouveau',
    'alphonse-mucha': 'Art Nouveau',
    
    # Japanese
    'katsushika-hokusai': 'Ukiyo-e (Japanese)',
    'utagawa-hiroshige': 'Ukiyo-e (Japanese)',
    
    # Folk/Naive
    'nicholas-roerich': 'Symbolism',
    'boris-kustodiev': 'Russian Realism',
    'konstantin-somov': 'Symbolism',
}

def extract_artist(filename):
    """Extract artist name from filename like 'claude-monet_the-cabin...'"""
    match = re.match(r'^([a-zA-Z-]+)_', filename)
    if match:
        return match.group(1).lower()
    return None

def get_style(artist):
    """Get art movement/style for an artist"""
    return ARTIST_STYLE_MAP.get(artist, 'Other')

def analyze_by_genre_style(results_csv, output_dir="results/genre_analysis"):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading results from {results_csv}...")
    df = pd.read_csv(results_csv)
    
    # Extract artist from filename
    df['artist'] = df['filename'].apply(extract_artist)
    df['style'] = df['artist'].apply(get_style)
    
    print(f"Total samples: {len(df)}")
    print(f"Unique artists: {df['artist'].nunique()}")
    print(f"Art styles identified: {df['style'].nunique()}")
    
    # ================================
    # 1. Style-wise Analysis
    # ================================
    print("\n" + "="*60)
    print("1. ANALYSIS BY ART MOVEMENT/STYLE")
    print("="*60)
    
    style_stats = df.groupby(['style', 'label']).agg({
        'creativity_score': ['mean', 'std', 'count'],
        'similarity': 'mean',
        'lpips': 'mean'
    }).round(4)
    
    print(style_stats)
    
    # Create summary table for styles with enough samples
    style_summary = []
    for style in df['style'].unique():
        style_df = df[df['style'] == style]
        ai_scores = style_df[style_df['label'] == 'ai']['creativity_score']
        human_scores = style_df[style_df['label'] == 'human']['creativity_score']
        
        if len(ai_scores) >= 5 and len(human_scores) >= 5:
            t_stat, p_val = stats.ttest_ind(ai_scores, human_scores, equal_var=False)
            style_summary.append({
                'Style': style,
                'AI_Mean': ai_scores.mean(),
                'Human_Mean': human_scores.mean(),
                'Diff': ai_scores.mean() - human_scores.mean(),
                'AI_Count': len(ai_scores),
                'Human_Count': len(human_scores),
                'p_value': p_val,
                'Significant': '*' if p_val < 0.05 else ''
            })
    
    style_summary_df = pd.DataFrame(style_summary).sort_values('Diff', ascending=False)
    print("\n[Style Summary (n≥5)]")
    print(style_summary_df.to_string(index=False))
    
    # ================================
    # 2. Artist-wise Analysis
    # ================================
    print("\n" + "="*60)
    print("2. ANALYSIS BY ARTIST (TOP 20)")
    print("="*60)
    
    artist_counts = df.groupby('artist').size().sort_values(ascending=False)
    top_artists = artist_counts.head(40).index.tolist()
    
    artist_summary = []
    for artist in top_artists:
        artist_df = df[df['artist'] == artist]
        ai_scores = artist_df[artist_df['label'] == 'ai']['creativity_score']
        human_scores = artist_df[artist_df['label'] == 'human']['creativity_score']
        
        if len(ai_scores) >= 1 and len(human_scores) >= 1:
            artist_summary.append({
                'Artist': artist,
                'Style': get_style(artist),
                'AI_Mean': ai_scores.mean(),
                'Human_Mean': human_scores.mean(),
                'Diff': ai_scores.mean() - human_scores.mean(),
                'Samples': len(ai_scores)
            })
    
    artist_summary_df = pd.DataFrame(artist_summary).sort_values('Diff', ascending=False)
    print(artist_summary_df.head(20).to_string(index=False))
    
    # ================================
    # 3. Visualizations
    # ================================
    print("\n" + "="*60)
    print("3. GENERATING VISUALIZATIONS")
    print("="*60)
    
    # 3.1 Style comparison bar chart
    plt.figure(figsize=(14, 8))
    style_means = df.groupby(['style', 'label'])['creativity_score'].mean().unstack()
    style_means = style_means.loc[style_means.sum(axis=1).sort_values(ascending=False).index]
    
    x = np.arange(len(style_means))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 8))
    bars1 = ax.bar(x - width/2, style_means['ai'], width, label='AI', color='red', alpha=0.7)
    bars2 = ax.bar(x + width/2, style_means['human'], width, label='Human', color='blue', alpha=0.7)
    
    ax.set_xlabel('Art Movement/Style')
    ax.set_ylabel('Mean Creativity Score')
    ax.set_title('Creativity Score by Art Movement (AI vs Human)')
    ax.set_xticks(x)
    ax.set_xticklabels(style_means.index, rotation=45, ha='right')
    ax.legend()
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'style_comparison.png'), dpi=150)
    plt.close()
    print("  Saved: style_comparison.png")
    
    # 3.2 Style boxplot
    plt.figure(figsize=(16, 8))
    styles_with_data = df.groupby('style').filter(lambda x: len(x) >= 10)['style'].unique()
    filtered_df = df[df['style'].isin(styles_with_data)]
    
    sns.boxplot(data=filtered_df, x='style', y='creativity_score', hue='label',
                palette={'ai': 'red', 'human': 'blue'})
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Art Movement/Style')
    plt.ylabel('Creativity Score')
    plt.title('Creativity Score Distribution by Style')
    plt.legend(title='Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'style_boxplot.png'), dpi=150)
    plt.close()
    print("  Saved: style_boxplot.png")
    
    # 3.3 Heatmap of AI-Human difference by style
    pivot = df.pivot_table(values='creativity_score', 
                           index='style', 
                           columns='label', 
                           aggfunc='mean')
    pivot['Diff'] = pivot['ai'] - pivot['human']
    pivot = pivot.sort_values('Diff', ascending=False)
    
    plt.figure(figsize=(10, 12))
    sns.heatmap(pivot[['Diff']], annot=True, cmap='RdBu_r', center=0, fmt='.3f')
    plt.title('AI - Human Creativity Score Difference by Style')
    plt.xlabel('Difference (positive = AI higher)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'style_diff_heatmap.png'), dpi=150)
    plt.close()
    print("  Saved: style_diff_heatmap.png")
    
    # 3.4 Artist comparison for top artists
    if len(artist_summary_df) >= 10:
        plt.figure(figsize=(14, 8))
        top_10 = artist_summary_df.head(10)
        bottom_10 = artist_summary_df.tail(10)
        combined = pd.concat([top_10, bottom_10])
        
        colors = ['red' if d > 0 else 'blue' for d in combined['Diff']]
        plt.barh(combined['Artist'], combined['Diff'], color=colors, alpha=0.7)
        plt.xlabel('Difference (AI - Human)')
        plt.ylabel('Artist')
        plt.title('AI vs Human Creativity Score Difference by Artist\n(Positive = AI higher error)')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'artist_diff_bar.png'), dpi=150)
        plt.close()
        print("  Saved: artist_diff_bar.png")
    
    # ================================
    # 4. Generate Report
    # ================================
    report_path = os.path.join(output_dir, 'genre_style_report.md')
    with open(report_path, 'w') as f:
        f.write("# Genre and Style Breakdown Analysis\n\n")
        f.write(f"**Analysis Date**: 2025-12-18\n")
        f.write(f"**Total Samples**: {len(df)}\n")
        f.write(f"**Unique Artists**: {df['artist'].nunique()}\n")
        f.write(f"**Art Movements**: {df['style'].nunique()}\n\n")
        
        f.write("## Art Movement Summary\n\n")
        f.write("| Style | AI Mean | Human Mean | Diff | Samples | p-value |\n")
        f.write("|-------|---------|------------|------|---------|--------|\n")
        for _, row in style_summary_df.iterrows():
            f.write(f"| {row['Style']} | {row['AI_Mean']:.3f} | {row['Human_Mean']:.3f} | {row['Diff']:+.3f} | {row['AI_Count']} | {row['p_value']:.4f}{row['Significant']} |\n")
        
        f.write("\n## Key Observations\n\n")
        
        # Find styles where AI > Human significantly
        ai_higher = style_summary_df[(style_summary_df['Diff'] > 0.1) & (style_summary_df['p_value'] < 0.1)]
        human_higher = style_summary_df[(style_summary_df['Diff'] < -0.1) & (style_summary_df['p_value'] < 0.1)]
        
        if len(ai_higher) > 0:
            f.write("### Styles where AI shows HIGHER reconstruction error:\n")
            for _, row in ai_higher.iterrows():
                f.write(f"- **{row['Style']}**: AI={row['AI_Mean']:.3f}, Human={row['Human_Mean']:.3f} (diff={row['Diff']:+.3f})\n")
            f.write("\n")
        
        if len(human_higher) > 0:
            f.write("### Styles where Human shows HIGHER reconstruction error:\n")
            for _, row in human_higher.iterrows():
                f.write(f"- **{row['Style']}**: AI={row['AI_Mean']:.3f}, Human={row['Human_Mean']:.3f} (diff={row['Diff']:+.3f})\n")
            f.write("\n")
        
        f.write("## Visualizations\n\n")
        f.write("![Style Comparison](style_comparison.png)\n")
        f.write("![Style Boxplot](style_boxplot.png)\n")
        f.write("![Style Difference Heatmap](style_diff_heatmap.png)\n")
        f.write("![Artist Difference](artist_diff_bar.png)\n")
    
    print(f"\n  Report saved to: {report_path}")
    
    # Save detailed CSV
    style_summary_df.to_csv(os.path.join(output_dir, 'style_summary.csv'), index=False)
    artist_summary_df.to_csv(os.path.join(output_dir, 'artist_summary.csv'), index=False)
    print("  Saved: style_summary.csv, artist_summary.csv")
    
    print("\nGenre/Style analysis complete!")
    
    return style_summary_df, artist_summary_df

if __name__ == "__main__":
    analyze_by_genre_style("results/wikiart_sd_full_scores.csv")
