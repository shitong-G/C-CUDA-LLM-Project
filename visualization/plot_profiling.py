import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style for professional looking plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

# Create output directory for figures if it doesn't exist
os.makedirs('visualization/figures', exist_ok=True)

def plot_kernel_breakdown():
    # Load data
    df = pd.read_csv('visualization/baseline_profile.csv')
    
    # Categorize kernels
    def categorize(name):
        name = name.lower()
        if 'gemm' in name or 'matmul' in name or 'cutlass' in name:
            return 'Matrix Multiplication (GEMM)'
        elif 'adamw' in name:
            return 'Optimizer (AdamW)'
        elif 'attention' in name or 'softmax' in name or 'permute' in name:
            return 'Attention Overhead'
        elif 'gelu' in name or 'layernorm' in name or 'residual' in name:
            return 'Activations & Norms'
        else:
            return 'Other'

    df['Category'] = df['Name'].apply(categorize)
    
    # Group by category
    category_time = df.groupby('Category')['Total Time (ns)'].sum().sort_values(ascending=False)
    
    # Calculate percentage
    total_time = category_time.sum()
    percentages = category_time / total_time * 100
    
    # Define colors
    colors = sns.color_palette('pastel')
    
    # Plot Pie Chart
    plt.figure(figsize=(12, 8))
    
    # Explode the largest slice (Matmul) slightly
    explode = [0.05 if i == 0 else 0 for i in range(len(category_time))]
    
    patches, texts, autotexts = plt.pie(
        category_time, 
        labels=None, # We'll add a legend instead
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        explode=explode,
        shadow=True,
        pctdistance=0.85
    )
    
    # Add a circle at the center to transform it into a donut chart
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    
    # Customize text
    for text in texts:
        text.set_fontsize(12)
    for autotext in autotexts:
        autotext.set_fontsize(14)
        autotext.set_weight('bold')
        
    plt.legend(
        category_time.index, 
        title="Kernel Categories", 
        loc="center left", 
        bbox_to_anchor=(1, 0, 0.5, 1)
    )
    
    plt.title('GPU Time Breakdown: FP32 Baseline\n(Where does the time go?)', fontsize=20, pad=20)
    plt.tight_layout()
    
    # Save
    output_path = 'visualization/figures/baseline_breakdown.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Pie chart saved to {output_path}")

def plot_throughput_baseline():
    # Data
    throughput = 5868  # From your run log (tokens/sec)
    
    plt.figure(figsize=(8, 6))
    
    # Create a single bar
    bars = plt.bar(['FP32 Baseline'], [throughput], color=['#4c72b0'], width=0.4)
    
    # Add value on top
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\ntokens/s',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    plt.title('Training Throughput Baseline\n(Higher is Better)', fontsize=16, pad=20)
    plt.ylabel('Throughput (tokens/s)', fontsize=14)
    
    # Set y-axis to start from 0 and give some headroom
    plt.ylim(0, 8000)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save
    output_path = 'visualization/figures/baseline_throughput.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Throughput chart saved to {output_path}")

if __name__ == "__main__":
    print("Generating plots...")
    plot_kernel_breakdown()
    plot_throughput_baseline()
    print("Done!")

