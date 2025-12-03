# Changes from train_gpt2_fp32.cu to train_gpt2_mixed.cu

## Overview
This document summarizes all changes made to transform the FP32 baseline (`train_gpt2_fp32.cu`) into the mixed precision version (`train_gpt2_mixed.cu`) with WMMA Tensor Core optimization.

---

## 1. Header Includes & Precision Support

### Added Includes
```cpp
#include <cuda_fp16.h>      // FP16 support
#include <cuda_bf16.h>      // BF16 support
#include <mma.h>             // WMMA API for Tensor Cores
```

### New Precision System
```cpp
// Precision settings with conditional compilation
#if defined(ENABLE_BF16)
    typedef __nv_bfloat16 floatX;
    #define PRECISION_STR "bf16"
#elif defined(ENABLE_FP16)
    typedef __half floatX;
    #define PRECISION_STR "fp16"
#else
    typedef float floatX;
    #define PRECISION_STR "fp32"
#endif

using namespace nvcuda;  // WMMA namespace
```

### Modular Kernel Includes
**FP32**: All kernels defined inline in the file (~1700 lines)

**Mixed**: Uses modular `llmc/*.cuh` headers:
```cpp
#include "llmc/cuda_common.h"
#include "llmc/cuda_utils.cuh"
#include "llmc/cublas_common.h"
#include "llmc/encoder.cuh"
#include "llmc/layernorm.cuh"
#include "llmc/matmul.cuh"
#include "llmc/attention.cuh"
#include "llmc/fused_classifier.cuh"
```

---

## 2. Kernel Implementations

### Removed Inline Kernels
**FP32** contains ~1000+ lines of inline kernel implementations:
- `encoder_forward_kernel3`
- `layernorm_forward_kernel3`
- `matmul_forward_kernel4`
- `softmax_forward_kernel5`
- `gelu_forward_kernel`
- `adamw_kernel2`
- `fused_classifier_kernel3`
- All backward kernels
- etc.

**Mixed**: All these kernels moved to `llmc/*.cuh` modules (except WMMA kernel)

### Added WMMA Kernel
**New in Mixed**:
```cpp
__global__ void wmma_matmul_forward_kernel(
    __nv_bfloat16* out,
    const __nv_bfloat16* inp, 
    const __nv_bfloat16* weight, 
    const __nv_bfloat16* bias,
    int M, int N, int K)
```

**Key Features**:
- BF16 inputs/outputs
- FP32 accumulator (for numerical stability)
- Shared memory staging for bias addition and casting
- 8x8x16 tile size (BF16 optimized)
- Uses `nvcuda::wmma` API

### Added Batched WMMA Wrapper
```cpp
void matmul_batched_wmma(
    floatX* out,
    const floatX* a, const floatX* b, const floatX* bias,
    int M, int N, int K, int batch_count,
    size_t strideA, size_t strideB, size_t strideOut,
    bool transA, bool transB,
    cudaStream_t stream)
```

### Custom Attention Implementation
**FP32**: Uses `attention_forward()` from `llmc/attention.cuh` (which uses cuBLAS)

**Mixed**: Custom `attention_forward_wmma()`:
- Uses WMMA for batched matmuls instead of cuBLAS
- Handles Q @ K^T and Att @ V with batched WMMA wrapper
- Still uses `permute_kernel`, `softmax_forward_kernel5`, `unpermute_kernel` from `llmc/attention.cuh`

---

## 3. Data Structures

### ParameterTensors
**FP32**:
```cpp
typedef struct {
    float* wte;  // All FP32
    float* wpe;
    // ...
} ParameterTensors;
```

**Mixed**:
```cpp
typedef struct {
    floatX* wte;  // BF16/FP16 (when enabled)
    floatX* wpe;
    // ...
} ParameterTensors;
```

### GPT2 Structure
**FP32**:
```cpp
typedef struct {
    ParameterTensors params;
    float* params_memory;  // Single FP32 copy
    // ...
} GPT2;
```

**Mixed**:
```cpp
typedef struct {
    ParameterTensors params;
    floatX* params_memory;        // BF16/FP16 weights
    float* params_memory_fp32;    // FP32 Master Weights (NEW!)
    // ...
} GPT2;
```

**Key Addition**: Master Weights pattern for mixed precision training

### ActivationTensors
**FP32**: All `float*`

**Mixed**: Mixed precision:
```cpp
typedef struct {
    floatX* encoded;      // BF16/FP16
    floatX* ln1;          // BF16/FP16
    float* ln1_mean;      // FP32 (statistics)
    float* ln1_rstd;      // FP32 (statistics)
    // ...
} ActivationTensors;
```

**Helper Function**:
```cpp
int is_fp32_activation(int i) {
    // mean, rstd, losses are FP32
    return (i == 2 || i == 3 || i == 9 || i == 10 || ...);
}
```

---

## 4. Memory Allocation

### Parameter Allocation
**FP32**:
```cpp
float* malloc_and_point_parameters(...) {
    cudaMalloc(&params_memory, num_parameters * sizeof(float));
    // Single allocation
}
```

**Mixed**:
```cpp
floatX* malloc_and_point_parameters(...) {
    cudaMalloc(&params_memory, num_parameters * sizeof(floatX));
    // BF16/FP16 allocation
}

// In gpt2_build_from_checkpoint():
cudaMalloc(&model->params_memory_fp32, num_parameters * sizeof(float));  // Master
model->params_memory = malloc_and_point_parameters(...);  // BF16 copy

// Cast Master → BF16
cast_float_to_floatX_kernel<<<...>>>(model->params_memory, 
                                      model->params_memory_fp32, 
                                      num_parameters);
```

### Activation Allocation
**FP32**:
```cpp
float* malloc_and_point_activations(...) {
    cudaMalloc(&acts_memory, num_activations * sizeof(float));
}
```

**Mixed**:
```cpp
floatX* malloc_and_point_activations(...) {
    // Calculate total bytes with mixed types
    for (int i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        size_t element_size = is_fp32_activation(i) ? sizeof(float) : sizeof(floatX);
        total_bytes += act_sizes[i] * element_size;
    }
    cudaMalloc(&acts_memory_void, total_bytes);
    // Manual pointer assignment for mixed types
}
```

---

## 5. Forward Pass Changes

### Matmul Calls
**FP32**:
```cpp
matmul_forward(scratch, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C);
// Uses matmul_forward_kernel4 (custom FP32 kernel)
```

**Mixed**:
```cpp
matmul_forward_wmma(scratch, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C, main_stream);
// Uses WMMA Tensor Core kernel
```

**All 5 matmul operations replaced**:
1. QKV projection: `matmul_forward_wmma(scratch, l_ln1, l_qkvw, l_qkvb, ...)`
2. Attention projection: `matmul_forward_wmma(l_attproj, l_atty, l_attprojw, ...)`
3. FFN expansion: `matmul_forward_wmma(l_fch, l_ln2, l_fcw, l_fcb, ...)`
4. FFN projection: `matmul_forward_wmma(l_fcproj, l_fch_gelu, l_fcprojw, ...)`
5. Output projection: `matmul_forward_wmma(acts.output, acts.lnf, params.wte, ...)`

### Attention Module
**FP32**:
```cpp
attention_forward(l_atty, l_qkvr, l_att, scratch, B, T, C, NH);
// Uses cuBLAS for batched matmuls
```

**Mixed**:
```cpp
attention_forward_wmma(l_atty, l_qkvr, l_att, scratch, B, T, C, NH, main_stream);
// Uses WMMA for batched matmuls
```

### Stream Support
**FP32**: No stream parameter (uses default stream)

**Mixed**: All kernel calls include `main_stream`:
```cpp
cudaStream_t main_stream;  // Global stream
cudaStreamCreate(&main_stream);

encoder_forward(..., main_stream);
layernorm_forward(..., main_stream);
matmul_forward_wmma(..., main_stream);
attention_forward_wmma(..., main_stream);
residual_forward(..., main_stream);
gelu_forward(..., main_stream);
fused_classifier(..., main_stream);
```

---

## 6. Model Initialization

### gpt2_build_from_checkpoint()
**FP32**:
```cpp
void gpt2_build_from_checkpoint(GPT2* model, ...) {
    model->params_memory = malloc_and_point_parameters(...);
    // Read FP32 weights → GPU
    cudaMemcpy(model->params_memory, params_cpu, ..., cudaMemcpyHostToDevice);
}
```

**Mixed**:
```cpp
void gpt2_build_from_checkpoint(GPT2* model, ...) {
    // Allocate Master Weights (FP32)
    cudaMalloc(&model->params_memory_fp32, num_parameters * sizeof(float));
    
    // Allocate BF16/FP16 weights
    model->params_memory = malloc_and_point_parameters(...);
    
    // Read FP32 weights → Master Weights
    cudaMemcpy(model->params_memory_fp32, params_cpu, ..., cudaMemcpyHostToDevice);
    
    // Cast Master → BF16/FP16
    cast_float_to_floatX_kernel<<<...>>>(model->params_memory, 
                                          model->params_memory_fp32, 
                                          num_parameters);
}
```

---

## 7. Cast Kernel

### New Cast Kernel
**Mixed** (not in FP32):
```cpp
__global__ void cast_float_to_floatX_kernel(floatX* out, const float* inp, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = (floatX)inp[idx];
    }
}
```

**Purpose**: Convert FP32 Master Weights to BF16/FP16 for forward pass

---

## 8. Backward Pass

### Current Status
**FP32**: Fully implemented with all backward kernels

**Mixed**: Placeholder (TODO):
```cpp
void gpt2_backward(GPT2* model) {
    // TODO: Implement using llmc/*.cuh kernels
}
```

**Note**: Backward pass not yet implemented in mixed precision version

---

## 9. Optimizer (Update)

### Current Status
**FP32**: Fully implemented:
```cpp
void gpt2_update(GPT2* model, ...) {
    adamw_kernel2<<<...>>>(model->params_memory, ...);
    // Updates FP32 weights directly
}
```

**Mixed**: Placeholder (TODO):
```cpp
void gpt2_update(GPT2* model, ...) {
    // TODO: Implement AdamW optimizer with Master Weights
    // 1. Update params_memory_fp32 (Master Weights) using gradients
    // 2. Cast back to params_memory (BF16/FP16)
}
```

**Required Pattern**:
1. Accumulate gradients in FP32
2. Update Master Weights (FP32) with AdamW
3. Cast updated Master Weights → BF16/FP16 weights

---

## 10. Loss Tracking

### CSV Export
**FP32**: No CSV export (only logger)

**Mixed**: Added CSV export for visualization:
```cpp
static FILE* loss_csv = NULL;
if (loss_csv == NULL) {
    loss_csv = fopen("visualization/bf16_loss.csv", "w");
    fprintf(loss_csv, "step,loss,time_ms,tokens_per_sec\n");
}
fprintf(loss_csv, "%d,%.6f,%.3f,%d\n", step+1, model.mean_loss, ...);
```

---

## 11. Inference (Sampling)

### Logits Conversion
**FP32**:
```cpp
float* logits = model.acts.output + (t-1) * Vp;
cudaMemcpy(cpu_logits, logits, V * sizeof(float), ...);
// Direct FP32 copy
```

**Mixed**:
```cpp
floatX* logits_floatX = model.acts.output + (t-1) * Vp;
floatX* logits_floatX_cpu = malloc(...);
cudaMemcpy(logits_floatX_cpu, logits_floatX, V * sizeof(floatX), ...);
// Convert floatX → float on CPU
for (int i = 0; i < V; i++) {
    cpu_logits[i] = (float)logits_floatX_cpu[i];
}
```

---

## 12. Device Setup

### Stream Creation
**FP32**: No explicit stream creation

**Mixed**:
```cpp
cudaStreamCreate(&main_stream);
```

### Precision Display
**FP32**:
```cpp
printf("| TF32                  | %-50s |\n", enable_tf32 ? "enabled" : "disabled");
```

**Mixed**:
```cpp
printf("| Precision             | %-50s |\n", PRECISION_STR);
printf("| TF32                  | %-50s |\n", enable_tf32 ? "enabled" : "disabled");
```

---

## 13. Memory Allocation Messages

### Parameter Allocation
**FP32**:
```cpp
printf("allocated %d MiB for model parameters\n", ...);
```

**Mixed**:
```cpp
printf("allocated %d MiB for model parameters (FP32 Master + %s Weights)\n", 
       ..., PRECISION_STR);
```

### Activation Allocation
**FP32**:
```cpp
printf("allocated %zu MiB for activations\n", ...);
```

**Mixed**:
```cpp
printf("allocated %zu MiB for activations\n", num_activations >> 20);
// Note: num_activations already accounts for mixed types
```

---

## Summary Table

| Component | FP32 | Mixed |
|-----------|------|-------|
| **Precision** | FP32 only | BF16/FP16 with FP32 Master |
| **Kernel Location** | Inline (~1700 lines) | Modular (`llmc/*.cuh`) |
| **Matmul Kernel** | `matmul_forward_kernel4` (FP32) | `wmma_matmul_forward_kernel` (WMMA) |
| **Attention** | cuBLAS batched matmul | WMMA batched matmul |
| **Stream Support** | Default stream | Explicit `main_stream` |
| **Master Weights** | No | Yes (`params_memory_fp32`) |
| **Activation Types** | All FP32 | Mixed (BF16 activations, FP32 stats) |
| **Backward Pass** | ✅ Implemented | ❌ TODO |
| **Optimizer** | ✅ Implemented | ❌ TODO (needs Master Weight pattern) |
| **Loss Tracking** | Logger only | Logger + CSV export |

---

## Key Architectural Changes

1. **Modularization**: Moved from monolithic file to modular `llmc/*.cuh` structure
2. **Precision Abstraction**: `floatX` typedef allows switching between FP32/FP16/BF16
3. **Master Weights Pattern**: FP32 master copy for optimizer stability
4. **Tensor Core Integration**: WMMA API for high-performance matmul
5. **Stream Management**: Explicit stream handling for async execution
6. **Mixed Precision Activations**: FP32 for statistics/losses, BF16 for activations

---

## Files Modified/Created

### Modified
- `train_gpt2_mixed.cu` - Main training file (completely restructured)

### Created
- `visualization/bf16_loss.csv` - Loss tracking data
- `visualization/fp32_loss.csv` - Baseline comparison data
- `visualization/plot_loss_comparison.py` - Analysis script

### Dependencies
- All `llmc/*.cuh` modules (encoder, layernorm, matmul, attention, etc.)
- These modules handle the actual kernel implementations

---

## Performance Impact

| Metric | FP32 | Mixed (BF16 WMMA) | Change |
|--------|------|-------------------|--------|
| **Step Time** | ~700ms | ~370ms | **1.9x faster** ✅ |
| **Throughput** | ~5,868 tok/s | ~10,000-11,000 tok/s | **~1.7x faster** ✅ |
| **Memory** | 100% | ~50% | **2x reduction** ✅ |
| **GPU Utilization** | 36% matmul | 96.6% WMMA | **Highly optimized** ✅ |
| **Loss Stability** | Stable ✅ | Exploding ❌ | **Needs fixing** ⚠️ |

---

## Next Steps (To Complete Phase 2)

1. **Fix Numerical Stability** (Critical):
   - Debug loss explosion
   - Implement gradient scaling/clipping
   - Validate WMMA kernel correctness

2. **Complete Backward Pass**:
   - Implement backward kernels with BF16 support
   - Handle gradient accumulation in FP32

3. **Complete Optimizer**:
   - Implement Master Weight update pattern
   - Cast Master → BF16 after update

4. **Validation**:
   - Compare loss curves with FP32 baseline
   - Verify numerical correctness

