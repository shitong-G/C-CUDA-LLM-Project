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

## Technical Stack

- **CUDA C/C++**: Core implementation language
- **WMMA API**: Tensor Core matrix operations
- **cuBLAS/cuBLASLt**: High-performance linear algebra (fallback)
- **Nsight Systems/Compute**: Profiling and analysis tools

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


## References

- **Base Repository**: [llm.c](https://github.com/karpathy/llm.c) by Andrej Karpathy
- **WMMA API**: [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)

---

## License

MIT (inherited from llm.c)
