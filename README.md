# C-CUDA-LLM-Project

**Course**: DD2360 Applied GPU Programming  
**Project**: Mixed Precision Training with Tensor Core Acceleration for GPT-2  
**Team**: Group 20 (Ruiling Li, Shitong Guo, Jiachen Shi)

---

## Project Overview

This project implements **mixed-precision training** with **Tensor Core acceleration** for GPT-2 using reduced precision (BF16) and the WMMA API. We aim to accelerate GPT-2 training by 2-4× while maintaining numerical stability through a hybrid precision strategy.

### Key Objectives

1. **Tensor Core Acceleration**: Replace FP32 matrix operations with BF16 WMMA kernels
2. **Mixed Precision Strategy**: Balance performance and numerical stability
3. **Performance Evaluation**: Measure speedup, memory reduction, and accuracy trade-offs

---

## Current Status

### ✅ Phase 2: Core Dev (Completed)

**Achievements:**
- ✅ Custom WMMA kernel implementation (`wmma_matmul_forward_kernel`)
- ✅ Full training pipeline (forward + backward + optimizer)
- ✅ **1.9× speedup** achieved (700ms → 370ms per step)
- ✅ **~50% memory reduction** (BF16 weights vs FP32)
- ✅ **96.6% GPU utilization** in WMMA kernels
- ✅ Master Weights pattern implemented (FP32 master, BF16 active)

**Key Discovery:**
- Loss divergence identified with aggressive BF16 quantization
- Root cause: Softmax/Attention operations require higher precision
- **Solution**: Hybrid precision strategy (Phase 3)

### 🎯 Phase 3: Fusion & Hybrid Precision (In Progress)

**Planned:**
- Hybrid precision: BF16 for Linear/MLP, FP32 for Attention
- Kernel fusion: GEMM + Bias + Activation
- Numerical stability validation

---

## Technical Stack

- **CUDA C/C++**: Core implementation language
- **WMMA API**: Tensor Core matrix operations
- **cuBLAS/cuBLASLt**: High-performance linear algebra (fallback)
- **Nsight Systems/Compute**: Profiling and analysis tools

---

## Project Structure

```
C-CUDA-LLM-Project/
├── train_gpt2_mixed.cu      # Main mixed-precision training file
├── docs/
│   └── dev_notes/
│       ├── DEV_LOG.md        # Detailed development log
│       └── CHANGES_FP32_TO_MIXED.md  # Architecture changes
└── visualization/            # Profiling data and plots
```

---

## Quick Start

### Build (Windows)

```powershell
nvcc train_gpt2_mixed.cu -o train_gpt2mixed.exe -O3 -I"dev" -DENABLE_BF16 -lcublas -lcublasLt -Xcompiler "/utf-8" --generate-code arch=compute_89,code=sm_89 -std=c++17
```

### Run

```powershell
.\train_gpt2mixed.exe
```

### Profiling

```powershell
nsys profile --trace=cuda,nvtx --output=benchmark_bf16 .\train_gpt2mixed.exe
```

---

## Key Results (Phase 2)

| Metric | FP32 Baseline | BF16 WMMA | Improvement |
|--------|---------------|-----------|-------------|
| **Step Time** | ~700ms | ~370ms | **1.9× faster** ✅ |
| **Throughput** | ~5,868 tok/s | ~10,000-11,000 tok/s | **~1.7× faster** ✅ |
| **Memory** | 100% | ~50% | **2× reduction** ✅ |
| **GPU Utilization** | 36% (matmul) | 96.6% (WMMA) | **Optimized** ✅ |
| **Training Stability** | Stable ✅ | Diverges ⚠️ | *Expected - Phase 3 fix* |

---

## Implementation Details

### Precision Strategy

- **Weights**: BF16 storage, FP32 master copy for optimizer
- **Activations**: BF16 for compute-intensive operations
- **Optimizer**: FP32 master weights updated, then cast to BF16

### WMMA Kernel

- **Tile Size**: 8×8×16 (BF16 optimized for RTX 4070)
- **Strategy**: BF16 inputs → FP32 accumulator → BF16 output
- **Integration**: All 5 linear layers + Attention module

### Training Pipeline

- **Forward**: WMMA kernels for all matmul operations
- **Backward**: BF16 gradient computation with cuBLASLt fallback
- **Optimizer**: AdamW with Master Weights pattern

---

## Timeline

| Phase | Date Range | Status | Focus |
|-------|------------|--------|-------|
| **1. Setup** | 11.20–11.27 | ✅ Complete | Baseline profiling |
| **2. Core Dev** | 11.28–12.07 | ✅ Complete | WMMA implementation |
| **3. Fusion & Hybrid Precision** | 12.08–12.17 | 🎯 In Progress | Mixed precision strategy |
| **4. Analysis** | 12.18–12.23 | ⏳ Pending | Performance evaluation |
| **5. Optimization** | 12.24–12.29 | ⏳ Pending | Fine-tuning |
| **6. Final** | 12.30–01.04 | ⏳ Pending | Report & presentation |

---

## Evaluation Plan

### Metrics

- **Performance**: Throughput, step time, kernel speedup
- **Efficiency**: Tensor Core utilization, memory bandwidth
- **Stability**: Loss convergence, numerical accuracy

### Platforms

- **Primary**: RTX 4070 Laptop GPU (Windows 11)
- **Secondary**: T4 GPU (Google Colab) - planned

### Test Configuration

- **Model**: GPT-2 (124M parameters)
- **Dataset**: tinyshakespeare
- **Task**: Pre-training loop (100-500 steps)

---

## Documentation

- **Development Log**: [`docs/dev_notes/DEV_LOG.md`](docs/dev_notes/DEV_LOG.md)
- **Architecture Changes**: [`docs/dev_notes/CHANGES_FP32_TO_MIXED.md`](docs/dev_notes/CHANGES_FP32_TO_MIXED.md)

---

## References

- **Base Repository**: [llm.c](https://github.com/karpathy/llm.c) by Andrej Karpathy
- **WMMA API**: [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)

---

## License

MIT (inherited from llm.c)
