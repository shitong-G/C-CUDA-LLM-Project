# Profiling Visualization Guide

## 概述

本项目包含两个profiling可视化脚本：
- `plot_profiling.py` - FP32 Baseline版本
- `plot_mixed_profiling.py` - Mixed Precision (Hybrid)版本

## 快速开始

### 方法1: 使用已知数据（推荐）

如果你已经有timeline profiling结果，可以直接运行：

```powershell
cd visualization
python plot_mixed_profiling.py
```

脚本会自动使用已知的Phase 3 profiling数据（来自timeline分析）：
- WMMA Matmul: 79.8%
- WMMA Matmul (Fused): 6.7%
- Conversion (BF16→FP32): 4.2%
- Conversion (FP32→BF16): 4.2%
- Softmax (FP32): 3.0%
- Classifier: 1.0%
- Other: 1.1%

### 方法2: 从nsys导出CSV数据

如果你想使用实际的profiling数据：

#### Step 1: 运行nsys profiling

```powershell
nsys profile --trace=cuda,nvtx --output=hybrid_precision --force-overwrite=true .\train_gpt2mixed.exe
```

#### Step 2: 导出CSV数据

在Nsight Systems GUI中：
1. 打开 `hybrid_precision.nsys-rep`
2. 点击 "Kernel Summary" 标签
3. 右键点击表格 → "Export" → "CSV"
4. 保存为 `visualization/mixed_profile.csv`

**CSV格式要求**:
- 必须包含 `Name` 列（kernel名称）
- 必须包含 `Time (%)` 或 `Total Time (ns)` 列
- 示例格式：
```csv
Time (%),Total Time (ns),Instances,Avg (ns),Name
79.8,7980000000,1000,7980000.0,wmma_matmul_forward_kernel
6.7,670000000,100,6700000.0,wmma_matmul_gelu_forward_kernel
...
```

#### Step 3: 运行可视化脚本

```powershell
cd visualization
python plot_mixed_profiling.py
```

脚本会自动检测 `mixed_profile.csv` 并使用它，如果不存在则使用已知数据。

## 生成的图表

运行 `plot_mixed_profiling.py` 会生成以下图表：

1. **`mixed_breakdown.png`** - Mixed Precision版本的kernel时间分布（Donut Chart）
2. **`comparison_baseline_vs_mixed.png`** - FP32 vs Mixed Precision对比
3. **`throughput_comparison.png`** - 吞吐量对比（tokens/s）
4. **`kernel_time_comparison.png`** - Kernel时间分布对比（Bar Chart）

所有图表保存在 `visualization/figures/` 目录。

## 自定义分类

如果需要修改kernel分类逻辑，编辑 `plot_mixed_profiling.py` 中的 `categorize_mixed()` 函数：

```python
def categorize_mixed(name):
    name = name.lower()
    if 'wmma' in name or 'matmul' in name:
        # ... 你的分类逻辑
    # ...
```

## 已知数据来源

当前使用的已知数据来自：
- **Phase 3 Profiling Analysis** (DEV_LOG.md)
- **Timeline Analysis** (nsys profile结果)
- **Kernel Time Distribution**: 
  - WMMA Matmul: 79.8% (wmma_matmul_forward_kernel)
  - WMMA Matmul (Fused): 6.7% (wmma_matmul_gelu_forward_kernel)
  - Conversion: 8.4% (4.2% × 2)
  - Softmax (FP32): 3.0%
  - Other: ~2%

## 故障排除

### 问题: CSV文件找不到

**解决方案**: 脚本会自动使用已知数据，无需CSV文件。

### 问题: 图表显示不正确

**检查**:
1. CSV格式是否正确（必须有 `Name` 和 `Time (%)` 或 `Total Time (ns)` 列）
2. Kernel名称是否匹配分类规则
3. 数据百分比是否合理（总和应接近100%）

### 问题: 导入错误

**确保安装依赖**:
```powershell
pip install pandas matplotlib seaborn numpy
```

## 对比FP32 Baseline

要生成FP32 baseline的可视化：

```powershell
cd visualization
python plot_profiling.py
```

这会生成：
- `baseline_breakdown.png`
- `baseline_throughput.png`

## 下一步

- 对比两个版本的图表，分析性能改进
- 识别性能瓶颈（如转换kernels开销）
- 评估Tensor Core利用率
- 分析内存带宽使用

