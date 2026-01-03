# Phase 3 Hybrid Precision Profiling Guide

## 目标
对比 Phase 2 (全BF16) vs Phase 3 (Hybrid Precision) 的性能差异，验证FP32 Softmax转换的性能开销。

## Profiling 命令

### 1. Phase 3 Hybrid Precision Profiling

```powershell
# 使用 nsys 进行 profiling
nsys profile --trace=cuda,nvtx --output=hybrid_precision --force-overwrite=true .\train_gpt2mixed.exe
```

这将生成：
- `hybrid_precision.nsys-rep` - 交互式报告
- `hybrid_precision.sqlite` - 数据库文件

### 2. 对比分析

需要对比的数据：
- **Phase 2 (全BF16)**: 已有的 `baseline_fp32.nsys-rep` 或 Phase 2 profile
- **Phase 3 (Hybrid Precision)**: 新生成的 `hybrid_precision.nsys-rep`

## 关键指标分析

### 1. Step Time 对比
- Phase 2: ~370ms
- Phase 3: 预期 ~390ms (增加转换开销)

### 2. Kernel 时间分布

需要关注的新kernels：
- `cast_bf16_to_float_kernel` - BF16→FP32转换
- `softmax_forward_kernel5_fp32` - FP32 Softmax
- `cast_float_to_bf16_kernel` - FP32→BF16转换

### 3. 性能开销分析

**预期开销来源**：
1. **转换kernels**: 2次转换 (BF16→FP32, FP32→BF16)
2. **FP32 Softmax**: 比BF16 Softmax稍慢但更稳定
3. **内存分配**: 静态FP32缓冲区分配

**预期结果**：
- Step time: 增加 ~5-10% (转换开销)
- Tensor Core利用率: 保持 ~96% (Linear/MLP仍用BF16 WMMA)
- 内存: 增加 ~B*NH*T*T*sizeof(float)*2 (临时FP32缓冲区)

## 分析步骤

### Step 1: 运行 Profiling
```powershell
# 确保使用最新的 Hybrid Precision 版本
.\build_windows.ps1 train_gpt2mixed

# 运行 profiling (限制步数以加快分析)
# 注意：需要修改训练循环以限制步数，或使用 Ctrl+C 中断
nsys profile --trace=cuda,nvtx --output=hybrid_precision --force-overwrite=true .\train_gpt2mixed.exe
```

### Step 2: 导出 Kernel 数据
在 Nsight Systems GUI 中：
1. 打开 `hybrid_precision.nsys-rep`
2. 导出 Timeline 数据
3. 查看 Kernel Summary
4. 导出 CSV 数据

### Step 3: 对比分析
对比以下指标：

| Metric | Phase 2 (全BF16) | Phase 3 (Hybrid) | 差异 |
|--------|------------------|------------------|------|
| Step Time | ~370ms | ? | ? |
| WMMA Kernel % | 96.6% | ? | ? |
| Softmax Time | 1.3% (BF16) | ? (FP32) | ? |
| Conversion Time | 0% | ? | ? |
| Memory Bandwidth | ? | ? | ? |

### Step 4: 识别性能瓶颈
- 转换kernels是否成为瓶颈？
- FP32 Softmax是否显著慢于BF16？
- 内存分配是否影响性能？

## 预期发现

### 性能开销
- **转换开销**: ~2-5% (BF16↔FP32转换)
- **FP32 Softmax**: 可能比BF16慢 ~10-20%
- **总体影响**: Step time 增加 ~5-10%

### 性能保持
- **Tensor Core利用率**: 应保持 ~96% (Linear/MLP仍用BF16 WMMA)
- **主要计算**: 仍由WMMA kernels主导
- **内存带宽**: 转换操作是内存bound，但开销较小

## 文档化结果

分析完成后，更新 `DEV_LOG.md` 的 Phase 3 部分，添加：
1. Hybrid Precision profiling 结果
2. 与 Phase 2 的对比
3. 性能开销分析
4. 转换kernels的性能特征


