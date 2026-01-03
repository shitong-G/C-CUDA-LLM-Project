import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Set style for professional looking plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, 'figures')
# Create output directory for figures if it doesn't exist
os.makedirs(FIGURES_DIR, exist_ok=True)

def plot_kernel_breakdown_mixed(csv_path=None):
    """
    Plot kernel breakdown for mixed precision (Phase 3 Hybrid Precision) version.
    
    If CSV doesn't exist, uses known profiling data from timeline analysis.
    """
    # Set default CSV path if not provided
    if csv_path is None:
        csv_path = os.path.join(SCRIPT_DIR, 'mixed_profile.csv')
    
    # Check if CSV exists, otherwise use known data
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        print(f"Warning: {csv_path} not found. Using known profiling data from timeline analysis.")
        print(f"  (Looking for CSV at: {csv_path})")
        # Create DataFrame from known Phase 3 profiling data
        # Data source: DEV_LOG.md Phase 3 Performance Analysis section (timeline profiling)
        # Based on nsys profile of hybrid_precision.nsys-rep
        # Total: 100.0% (verified: 79.8 + 6.7 + 4.2 + 4.2 + 3.0 + 1.0 + 1.1 = 100.0)
        data = {
            'Name': [
                'wmma_matmul_forward_kernel',           # 79.8% - Main WMMA matmul
                'wmma_matmul_gelu_forward_kernel',      # 6.7% - Fused WMMA+GELU
                'cast_bf16_to_float_kernel',            # 4.2% - BF16→FP32 conversion
                'cast_float_to_bf16_kernel',            # 4.2% - FP32→BF16 conversion
                'softmax_forward_kernel5_fp32',         # 3.0% - FP32 Softmax
                'fused_classifier_kernel5',             # 1.0% - Output classifier
                'Other kernels'                         # 1.1% - LayerNorm, residual, etc.
            ],
            'Time (%)': [79.8, 6.7, 4.2, 4.2, 3.0, 1.0, 1.1],
            'Total Time (ns)': [0, 0, 0, 0, 0, 0, 0]  # Will be calculated from percentages
        }
        df = pd.DataFrame(data)
        # Calculate total time (assuming 100% = some baseline)
        total_time_ns = 10000000000  # 10 seconds as baseline
        df['Total Time (ns)'] = df['Time (%)'] / 100.0 * total_time_ns
    
    # Categorize kernels for mixed precision
    def categorize_mixed(name):
        name = name.lower()
        if 'wmma' in name or 'matmul' in name:
            if 'gelu' in name:
                return 'WMMA Matmul (Fused)'
            else:
                return 'WMMA Matmul'
        elif 'cast' in name:
            if 'bf16_to_float' in name:
                return 'Conversion (BF16→FP32)'
            elif 'float_to_bf16' in name:
                return 'Conversion (FP32→BF16)'
            else:
                return 'Conversion'
        elif 'softmax' in name:
            if 'fp32' in name:
                return 'Softmax (FP32)'
            else:
                return 'Softmax'
        elif 'classifier' in name or 'fused_classifier' in name:
            return 'Classifier'
        elif 'layernorm' in name:
            return 'LayerNorm'
        elif 'gelu' in name:
            return 'GELU'
        else:
            return 'Other'
    
    # If CSV has 'Time (%)' column, use it; otherwise calculate from 'Total Time (ns)'
    if 'Time (%)' in df.columns:
        df['Category'] = df['Name'].apply(categorize_mixed)
        category_time = df.groupby('Category')['Time (%)'].sum().sort_values(ascending=False)
        # Convert to nanoseconds for plotting (normalize to 100%)
        total_time = category_time.sum()
        category_time_ns = category_time / total_time * 10000000000  # Normalize to 10s
    else:
        df['Category'] = df['Name'].apply(categorize_mixed)
        category_time_ns = df.groupby('Category')['Total Time (ns)'].sum().sort_values(ascending=False)
        total_time = category_time_ns.sum()
        category_time = category_time_ns / total_time * 100
    
    # Define colors for mixed precision categories
    color_map = {
        'WMMA Matmul': '#2ecc71',           # Green
        'WMMA Matmul (Fused)': '#27ae60',  # Darker green
        'Conversion (BF16→FP32)': '#e74c3c',  # Red
        'Conversion (FP32→BF16)': '#c0392b',  # Darker red
        'Softmax (FP32)': '#3498db',       # Blue
        'Classifier': '#9b59b6',           # Purple
        'LayerNorm': '#f39c12',            # Orange
        'GELU': '#1abc9c',                 # Teal
        'Other': '#95a5a6'                 # Gray
    }
    
    colors = [color_map.get(cat, '#95a5a6') for cat in category_time_ns.index]
    
    # Plot Donut Chart
    plt.figure(figsize=(14, 10))
    
    # Explode the largest slice (WMMA Matmul) slightly
    explode = [0.05 if 'WMMA' in str(cat) else 0 for cat in category_time_ns.index]
    
    patches, texts, autotexts = plt.pie(
        category_time_ns, 
        labels=None,
        autopct=lambda pct: f'{pct:.1f}%',
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
        # Make text white for dark colors
        if autotext.get_text() != '':
            autotext.set_color('white')
    
    plt.legend(
        [f'{cat}: {pct:.1f}%' for cat, pct in zip(category_time_ns.index, category_time)],
        title="Kernel Categories (Mixed Precision)",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
        fontsize=11
    )
    
    plt.title('GPU Time Breakdown: Mixed Precision (Hybrid)\nWMMA + FP32 Softmax + Conversions', 
              fontsize=20, pad=20)
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(FIGURES_DIR, 'mixed_breakdown.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to free memory
    print(f"Mixed precision pie chart saved to {output_path}")

def plot_comparison_baseline_vs_mixed():
    """
    Compare FP32 Baseline vs Mixed Precision side by side.
    """
    # Known data from profiling
    baseline_data = {
        'Matrix Multiplication (GEMM)': 70.0,
        'Optimizer (AdamW)': 16.0,
        'Attention Overhead': 8.0,
        'Activations & Norms': 6.0
    }
    
    mixed_data = {
        'WMMA Matmul': 79.8,
        'WMMA Matmul (Fused)': 6.7,
        'Conversion (BF16→FP32)': 4.2,
        'Conversion (FP32→BF16)': 4.2,
        'Softmax (FP32)': 3.0,
        'Classifier': 1.0,
        'Other': 1.1
    }
    
    # Aggregate mixed data for comparison
    mixed_aggregated = {
        'WMMA Matmul (Total)': 79.8 + 6.7,
        'Conversion (Total)': 4.2 + 4.2,
        'Softmax (FP32)': 3.0,
        'Other': 1.0 + 1.1
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Baseline plot
    colors_baseline = ['#4c72b0', '#55a868', '#c44e52', '#8172b2']
    ax1.pie(baseline_data.values(), labels=baseline_data.keys(), autopct='%1.1f%%',
            colors=colors_baseline, startangle=140, shadow=True)
    ax1.set_title('FP32 Baseline\n(700ms/step)', fontsize=16, pad=20)
    
    # Mixed precision plot
    colors_mixed = ['#2ecc71', '#e74c3c', '#3498db', '#95a5a6']
    ax2.pie(mixed_aggregated.values(), labels=mixed_aggregated.keys(), autopct='%1.1f%%',
            colors=colors_mixed, startangle=140, shadow=True)
    ax2.set_title('Mixed Precision (Hybrid)\n(390ms/step, 1.8× faster)', fontsize=16, pad=20)
    
    plt.suptitle('Performance Comparison: FP32 vs Mixed Precision', fontsize=20, y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(FIGURES_DIR, 'comparison_baseline_vs_mixed.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to free memory
    print(f"Comparison chart saved to {output_path}")

def plot_throughput_comparison():
    """
    Compare throughput between FP32 Baseline and Mixed Precision.
    """
    data = {
        'FP32 Baseline': 5868,
        'Mixed Precision (Hybrid)': 10000  # Approximate from profiling
    }
    
    colors = ['#4c72b0', '#2ecc71']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(data.keys(), data.values(), color=colors, width=0.5)
    
    # Add value on top
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\ntokens/s',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    plt.title('Training Throughput Comparison\n(Higher is Better)', fontsize=16, pad=20)
    plt.ylabel('Throughput (tokens/s)', fontsize=14)
    plt.ylim(0, 12000)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add speedup annotation
    speedup = data['Mixed Precision (Hybrid)'] / data['FP32 Baseline']
    plt.text(0.5, 0.95, f'{speedup:.2f}× speedup', 
             transform=plt.gca().transAxes,
             ha='center', fontsize=14, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    output_path = os.path.join(FIGURES_DIR, 'throughput_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to free memory
    print(f"Throughput comparison saved to {output_path}")

def plot_kernel_time_comparison():
    """
    Compare kernel time distribution between FP32 and Mixed Precision.
    """
    categories = ['Matmul/GEMM', 'Softmax', 'Conversion', 'Other']
    fp32_pct = [70.0, 3.2, 0.0, 26.8]  # From baseline_profile.csv
    mixed_pct = [86.5, 3.0, 8.4, 2.1]  # From Phase 3 profiling
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, fp32_pct, width, label='FP32 Baseline', color='#4c72b0')
    bars2 = ax.bar(x + width/2, mixed_pct, width, label='Mixed Precision', color='#2ecc71')
    
    ax.set_ylabel('Time (%)', fontsize=14)
    ax.set_title('Kernel Time Distribution Comparison', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    output_path = os.path.join(FIGURES_DIR, 'kernel_time_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to free memory
    print(f"Kernel time comparison saved to {output_path}")

if __name__ == "__main__":
    import numpy as np
    
    print("Generating mixed precision profiling plots...")
    plot_kernel_breakdown_mixed()
    plot_comparison_baseline_vs_mixed()
    plot_throughput_comparison()
    plot_kernel_time_comparison()
    print("Done!")

