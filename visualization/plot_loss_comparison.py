#!/usr/bin/env python3
"""
Plot comparison between FP32 Baseline and BF16 WMMA Optimization
Shows the speed-stability tradeoff in mixed precision training
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys

# === 数据读取 ===
# Script is in visualization/ directory, so use current directory
script_dir = os.path.dirname(os.path.abspath(__file__))
visualization_dir = script_dir  # Current directory is visualization/
figures_dir = os.path.join(visualization_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)

fp32_file = os.path.join(visualization_dir, "fp32_loss.csv")
bf16_file = os.path.join(visualization_dir, "bf16_loss.csv")

# 检查文件是否存在
if not os.path.exists(fp32_file):
    print(f"Warning: {fp32_file} not found. Using dummy data.")
    fp32_data = None
else:
    fp32_data = pd.read_csv(fp32_file)

if not os.path.exists(bf16_file):
    print(f"Warning: {bf16_file} not found. Using dummy data.")
    bf16_data = None
else:
    bf16_data = pd.read_csv(bf16_file)

# === 创建图表 ===
fig = plt.figure(figsize=(14, 8))

# === 图表 A: Loss 曲线对比 (双轴) ===
ax1 = plt.subplot(2, 1, 1)

if fp32_data is not None:
    steps_fp32 = fp32_data['step'].values
    loss_fp32 = fp32_data['loss'].values
    ax1.plot(steps_fp32, loss_fp32, color='tab:blue', linewidth=2.5, 
             label='FP32 Baseline (Stable)', marker='o', markersize=3, alpha=0.8)
else:
    # Dummy data for demonstration
    steps_fp32 = np.arange(1, 75)
    loss_fp32 = np.linspace(4.3, 3.4, 74) + np.random.normal(0, 0.05, 74)
    ax1.plot(steps_fp32, loss_fp32, color='tab:blue', linewidth=2.5, 
             label='FP32 Baseline (Stable)', marker='o', markersize=3, alpha=0.8)

ax1.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
ax1.set_ylabel('FP32 Training Loss', color='tab:blue', fontsize=12, fontweight='bold')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.set_ylim(0, max(loss_fp32) * 1.2)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(loc='upper left', fontsize=11)

# 右轴：BF16 Loss
ax2 = ax1.twinx()

if bf16_data is not None:
    steps_bf16 = bf16_data['step'].values
    loss_bf16 = bf16_data['loss'].values
    ax2.plot(steps_bf16, loss_bf16, color='tab:red', linewidth=2.5, 
             label='BF16 WMMA (Exploding)', marker='s', markersize=3, 
             linestyle='--', alpha=0.8)
else:
    # Dummy data for demonstration
    steps_bf16 = np.arange(1, 75)
    loss_bf16 = np.linspace(69.15, 816.56, 74)
    ax2.plot(steps_bf16, loss_bf16, color='tab:red', linewidth=2.5, 
             label='BF16 WMMA (Exploding)', marker='s', markersize=3, 
             linestyle='--', alpha=0.8)

ax2.set_ylabel('BF16 WMMA Training Loss', color='tab:red', fontsize=12, fontweight='bold')
ax2.tick_params(axis='y', labelcolor='tab:red')
ax2.legend(loc='upper right', fontsize=11)

# 添加标题
ax1.set_title('FP32 Baseline vs. BF16 WMMA Optimization\n(The Speed-Stability Tradeoff)', 
               fontsize=14, fontweight='bold', pad=20)

# 添加标注
if fp32_data is not None and bf16_data is not None:
    avg_time_fp32 = fp32_data['time_ms'].mean()
    avg_time_bf16 = bf16_data['time_ms'].mean()
    speedup = avg_time_fp32 / avg_time_bf16
    ax1.text(0.5, 0.95, f'Speedup: {speedup:.2f}x\n({avg_time_fp32:.0f}ms → {avg_time_bf16:.0f}ms/step)', 
             transform=ax1.transAxes, fontsize=11, 
             bbox=dict(facecolor='yellow', alpha=0.6, edgecolor='black', linewidth=1.5),
             ha='center', va='top', fontweight='bold')

# === 图表 B: 性能对比 (柱状图) ===
ax3 = plt.subplot(2, 1, 2)

if fp32_data is not None and bf16_data is not None:
    avg_time_fp32 = fp32_data['time_ms'].mean()
    avg_time_bf16 = bf16_data['time_ms'].mean()
    avg_tokens_fp32 = fp32_data['tokens_per_sec'].mean()
    avg_tokens_bf16 = bf16_data['tokens_per_sec'].mean()
    speedup = avg_time_fp32 / avg_time_bf16
    
    categories = ['Time per Step\n(ms)', 'Throughput\n(tokens/sec)']
    fp32_values = [avg_time_fp32, avg_tokens_fp32]
    bf16_values = [avg_time_bf16, avg_tokens_bf16]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, fp32_values, width, label='FP32 Baseline', 
                    color='tab:blue', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax3.bar(x + width/2, bf16_values, width, label='BF16 WMMA', 
                    color='tab:red', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 添加 speedup 标注
    ax3.text(0, max(fp32_values[0], bf16_values[0]) * 1.15, 
             f'{speedup:.2f}x Speedup!', fontsize=12, fontweight='bold',
             ha='center', bbox=dict(facecolor='green', alpha=0.3, edgecolor='green', linewidth=2))
    
    ax3.set_ylabel('Performance Metric', fontsize=12, fontweight='bold')
    ax3.set_title('Performance Comparison: Speedup Achieved', fontsize=13, fontweight='bold', pad=15)
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories, fontsize=11)
    ax3.legend(fontsize=11, loc='upper left')
    ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
else:
    # Dummy data
    categories = ['Time per Step\n(ms)', 'Throughput\n(tokens/sec)']
    fp32_values = [700, 6000]
    bf16_values = [370, 10000]
    speedup = fp32_values[0] / bf16_values[0]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, fp32_values, width, label='FP32 Baseline', 
                    color='tab:blue', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax3.bar(x + width/2, bf16_values, width, label='BF16 WMMA', 
                    color='tab:red', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax3.text(0, max(fp32_values[0], bf16_values[0]) * 1.15, 
             f'{speedup:.2f}x Speedup!', fontsize=12, fontweight='bold',
             ha='center', bbox=dict(facecolor='green', alpha=0.3, edgecolor='green', linewidth=2))
    
    ax3.set_ylabel('Performance Metric', fontsize=12, fontweight='bold')
    ax3.set_title('Performance Comparison: Speedup Achieved', fontsize=13, fontweight='bold', pad=15)
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories, fontsize=11)
    ax3.legend(fontsize=11, loc='upper left')
    ax3.grid(True, alpha=0.3, linestyle='--', axis='y')

plt.tight_layout()
output_file = os.path.join(figures_dir, "loss_comparison.png")
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Plot saved to {output_file}")
plt.show()

