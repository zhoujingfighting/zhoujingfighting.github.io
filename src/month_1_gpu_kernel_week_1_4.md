# Month 1 GPU Kernel 实战代码与命令行速查笔记（Week 1 – Week 4）
**硬件目标平台**：NVIDIA B200 (Blackwell, `sm_100`, 148 SMs, 126.5 MB L2 Cache)  
**工程目录**：`~/workspace/llm-gpu-compute-lab`

---

## 0. 全局工程配置与通用工具（`CMakeLists.txt` & `cuda_check.h`）

### 0.1 `CMakeLists.txt`（全月通用编译配置）
```cmake
cmake_minimum_required(VERSION 3.24)
project(llm_gpu_compute_lab VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# B200 设置为 100；若在 H100 上运行可改为 90
set(CMAKE_CUDA_ARCHITECTURES 100)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)

add_library(cuda_common INTERFACE)
target_include_directories(cuda_common INTERFACE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(cuda_common INTERFACE CUDA::cudart CUDA::cublas)

foreach(TARGET_NAME vector_add gemm norm_softmax attention_kv)
    if(TARGET_NAME STREQUAL "vector_add")
        set(SRC_FILE src/week01/vector_add.cu)
    elseif(TARGET_NAME STREQUAL "gemm")
        set(SRC_FILE src/week02/gemm.cu)
    elseif(TARGET_NAME STREQUAL "norm_softmax")
        set(SRC_FILE src/week03/norm_softmax.cu)
    elseif(TARGET_NAME STREQUAL "attention_kv")
        set(SRC_FILE src/week04/attention_kv.cu)
    endif()

    if(EXISTS "${PROJECT_SOURCE_DIR}/${SRC_FILE}")
        add_executable(${TARGET_NAME} ${SRC_FILE})
        target_link_libraries(${TARGET_NAME} PRIVATE cuda_common)
        target_compile_options(${TARGET_NAME} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --expt-relaxed-constexpr>
            $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>
        )
    endif()
endforeach()
```

### 0.2 `include/cuda_check.h`
```cpp
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA Error at %s:%d - %s (%s)\n",            \
                         __FILE__, __LINE__, #call, cudaGetErrorString(err));  \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t status = (call);                                        \
        if (status != CUBLAS_STATUS_SUCCESS) {                                 \
            std::fprintf(stderr, "cuBLAS Error at %s:%d - %s (code %d)\n",     \
                         __FILE__, __LINE__, #call, static_cast<int>(status)); \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)
```

### 0.3 全月通用编译、Profiling 与汇编导出命令行速查
```bash
# 1. CMake + Ninja Release 编译
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# 2. 检查目标架构是否生效
grep -n "gpu-architecture" build/build.ninja

# 3. Nsight Compute (NCU) 导出包含源码的完整报告（配合代码内 cudaProfilerStart/Stop）
mkdir -p reports
ncu --set full --import-source yes --profile-from-start off -f -o reports/<report_name> ./build/<target> [args]

# 4. 日常快速调优采样（仅 4~12 Passes，速度极快）
ncu --set detailed --profile-from-start off ./build/<target> [args]

# 5. Nsight Systems (NSYS) 采集时间线与 Kernel Launch 开销
nsys profile --trace=cuda,nvtx,osrt -f true -o reports/<timeline_name> ./build/<target> [args]
nsys stats reports/<timeline_name>.nsys-rep

# 6. 导出 PTX 与 SASS 汇编分析指令调度与寄存器分配
cuobjdump --dump-ptx  build/<target> > reports/<target>.ptx
cuobjdump --dump-sass build/<target> > reports/<target>.sass
```

---

## Week 1：CUDA 编程模型、合并访存与性能分析

### 1.1 核心代码（`src/week01/vector_add.cu` 核心算子）
```cuda
#include "cuda_check.h"
#include <cuda_profiler_api.h>

// 1. 合并访存 (Coalesced): 相邻线程访问连续 32B Sector，100% 总线利用率
__global__ void vector_add_coalesced_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    size_t N) {
    size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < N) {
        C[tid] = A[tid] + B[tid];
    }
}

// 2. 跨步访存 (Strided, stride=32): 同一 Warp 跨 32 个 Cache Line，有效带宽衰减约 8x
__global__ void vector_add_strided_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    size_t N,
    size_t stride) {
    size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < N) {
        size_t idx = (tid * stride) % N + (tid * stride) / N;
        C[idx] = A[idx] + B[idx];
    }
}

// 3. 向量化访存 (float4 / LDG.E.128): 提升单指令字节吞吐，缓解 B200 HBM3e 延迟停顿
__global__ void vector_add_float4_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    size_t N) {
    size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t idx4 = tid * 4;
    if (idx4 + 3 < N) {
        float4 a = reinterpret_cast<const float4*>(A)[tid];
        float4 b = reinterpret_cast<const float4*>(B)[tid];
        float4 c;
        c.x = a.x + b.x; c.y = a.y + b.y; c.z = a.z + b.z; c.w = a.w + b.w;
        reinterpret_cast<float4*>(C)[tid] = c;
    } else {
        for (size_t i = idx4; i < N; ++i) C[i] = A[i] + B[i];
    }
}
```

### 1.2 Week 1 实战命令行
```bash
# 跑全套 4 种规模 N (1024 ~ 16777216) x 4 种 BlockSize (128 ~ 1024)
./build/vector_add

# 对比 Cold L2 Cache 模式（消除片上 L2 缓存命中假象）
./build/vector_add --cold-l2

# 抓取 N=16777216, BS=256 的 NCU 报告（跳过 50 次设备级预热 + 20 次 Case 预热）
ncu --set full --import-source yes -f -o reports/b200_vecadd_coal --launch-skip 70 --launch-count 1 ./build/vector_add 16777216 256
ncu --set full --import-source yes -f -o reports/b200_vecadd_stride -k regex:strided --launch-skip 20 --launch-count 1 ./build/vector_add 16777216 256

# 观察小尺寸 N=1024 下的 Host Enqueue / Kernel Launch 开销
nsys profile --trace=cuda,osrt -f true -o reports/vecadd_launch_overhead ./build/vector_add 1024 256
```

---

## Week 2：GEMM 分级存储优化（Global $\rightarrow$ Shared $\rightarrow$ Register）

### 2.1 核心代码（`src/week02/gemm.cu` 核心算子）
```cuda
// 1. Naive GEMM: 算术强度 0.25 FLOP/Byte，严重受限于 Global Memory 带宽
__global__ void gemm_naive_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

// 2. Shared Memory Tiling (32x32): 减少 32 倍 Global Memory 访问
template <int TILE_SIZE = 32>
__global__ void gemm_smem_tiled_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    float sum = 0.0f;

    for (int t = 0; t < K; t += TILE_SIZE) {
        As[ty][tx] = (row < M && (t + tx) < K) ? A[row * K + t + tx] : 0.0f;
        Bs[ty][tx] = ((t + ty) < K && col < N) ? B[(t + ty) * N + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) sum += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = sum;
}

// 3. 2D Register Blocking + float4 (BM=128, BN=128, BK=8, TM=8, TN=8)
// 单线程维护 8x8=64 个累加寄存器，每 16 次 LDS 驱动 64 次 FFMA
template <int BM = 128, int BN = 128, int BK = 8, int TM = 8, int TN = 8>
__global__ void gemm_reg_tiled_float4_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int N, int K) {
    __shared__ float As[BK][BM + 4]; // +4 Padding 消除转置写 Bank Conflict
    __shared__ float Bs[BK][BN];

    int tid = threadIdx.x;
    int ty = tid / (BN / TN), tx = tid % (BN / TN);
    int load_a_row = tid / 2, load_a_col = (tid % 2) * 4;
    int load_b_row = tid / 32, load_b_col = (tid % 32) * 4;

    float accum[TM][TN] = {0.0f};
    float frag_a[TM], frag_b[TN];

    const float* A_base = A + blockIdx.y * BM * K;
    const float* B_base = B + blockIdx.x * BN;

    for (int k_tile = 0; k_tile < K; k_tile += BK) {
        float4 tmp_a = reinterpret_cast<const float4*>(&A_base[load_a_row * K + k_tile + load_a_col])[0];
        As[load_a_col + 0][load_a_row] = tmp_a.x;
        As[load_a_col + 1][load_a_row] = tmp_a.y;
        As[load_a_col + 2][load_a_row] = tmp_a.z;
        As[load_a_col + 3][load_a_row] = tmp_a.w;

        reinterpret_cast<float4*>(&Bs[load_b_row][load_b_col])[0] =
            reinterpret_cast<const float4*>(&B_base[(k_tile + load_b_row) * N + load_b_col])[0];
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) frag_a[i] = As[k][ty * TM + i];
            #pragma unroll
            for (int j = 0; j < TN; ++j) frag_b[j] = Bs[k][tx * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    accum[i][j] += frag_a[i] * frag_b[j];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int c_row = blockIdx.y * BM + ty * TM + i;
        #pragma unroll
        for (int j = 0; j < TN; j += 4) {
            int c_col = blockIdx.x * BN + tx * TN + j;
            float4 res = {accum[i][j], accum[i][j + 1], accum[i][j + 2], accum[i][j + 3]};
            reinterpret_cast<float4*>(&C[c_row * N + c_col])[0] = res;
        }
    }
}
```

### 2.2 Week 2 实战命令行
```bash
# 运行 M=N=K in {512, 1024, 2048, 4096} 全套对比（含 cuBLAS 基准）
./build/gemm

# 一次性将 cuBLAS、Naive、Smem Tiling、2D Reg Tiling 4 个 Kernel 抓入同一份报告
ncu --set full --import-source yes --profile-from-start off -f -o reports/b200_gemm_4096 ./build/gemm 4096

# 导出 SASS 检查 LDS.128 与 FFMA 指令交错排布及 Registers Per Thread
cuobjdump --dump-sass build/gemm > reports/gemm.sass
```

---

## Week 3：Softmax 与 LayerNorm / RMSNorm 算子融合

### 3.1 核心代码（`src/week03/norm_softmax.cu` 核心归约与融合算子）
```cuda
#include "cuda_check.h"
#include <cfloat>
#include <cmath>

// ============================================================================
// 1. Warp & Block 级并行归约原语 (基于 __shfl_xor_sync 寄存器直连交换)
// ============================================================================
__inline__ __device__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

template <int BLOCK_SIZE>
__inline__ __device__ float block_reduce_max(float val) {
    static __shared__ float shared[32]; // 最多 1024 / 32 = 32 个 Warp
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    val = warp_reduce_max(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < (BLOCK_SIZE >> 5)) ? shared[lane] : -FLT_MAX;
    if (wid == 0) val = warp_reduce_max(val);
    return val;
}

template <int BLOCK_SIZE>
__inline__ __device__ float block_reduce_sum(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    val = warp_reduce_sum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < (BLOCK_SIZE >> 5)) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    return val;
}

// ============================================================================
// 2. Online Softmax Kernel (单趟遍历完成 Max + Sum 归约，支持任意超大词表/长序列)
// ============================================================================
struct __align__(8) MD {
    float m; // 当前局部最大值
    float d; // 修正后的局部指数和
};

__inline__ __device__ MD reduce_md_op(MD a, MD b) {
    bool a_bigger = (a.m > b.m);
    MD bigger  = a_bigger ? a : b;
    MD smaller = a_bigger ? b : a;
    MD res;
    res.m = bigger.m;
    res.d = bigger.d + smaller.d * expf(smaller.m - bigger.m);
    return res;
}

template <int BLOCK_SIZE = 256>
__global__ void online_softmax_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_in  = x + static_cast<size_t>(row) * cols;
    float*       row_out = y + static_cast<size_t>(row) * cols;

    // Pass 1: 单次遍历计算线程局部 (m, d)
    MD thread_md = {-FLT_MAX, 0.0f};
    for (int col = threadIdx.x; col < cols; col += BLOCK_SIZE) {
        MD curr = {row_in[col], 1.0f};
        thread_md = reduce_md_op(thread_md, curr);
    }

    // Warp 内合并 (m, d)
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        MD other;
        other.m = __shfl_xor_sync(0xffffffff, thread_md.m, offset);
        other.d = __shfl_xor_sync(0xffffffff, thread_md.d, offset);
        thread_md = reduce_md_op(thread_md, other);
    }

    // 跨 Warp 合并
    __shared__ MD smem_md[32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    if (lane == 0) smem_md[wid] = thread_md;
    __syncthreads();

    if (wid == 0) {
        thread_md = (lane < (BLOCK_SIZE >> 5)) ? smem_md[lane] : MD{-FLT_MAX, 0.0f};
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            MD other;
            other.m = __shfl_xor_sync(0xffffffff, thread_md.m, offset);
            other.d = __shfl_xor_sync(0xffffffff, thread_md.d, offset);
            thread_md = reduce_md_op(thread_md, other);
        }
        if (lane == 0) smem_md[0] = thread_md;
    }
    __syncthreads();

    MD final_md = smem_md[0];
    float inv_d = 1.0f / final_md.d;

    // Pass 2: 写回归一化结果
    for (int col = threadIdx.x; col < cols; col += BLOCK_SIZE) {
        row_out[col] = expf(row_in[col] - final_md.m) * inv_d;
    }
}

// ============================================================================
// 3. Fused RMSNorm Kernel (LLM 主流归一化算子，支持 float4 向量化)
// ============================================================================
template <int BLOCK_SIZE = 256>
__global__ void fused_rmsnorm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ y,
    int rows, int cols, float eps) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_in  = x + static_cast<size_t>(row) * cols;
    float*       row_out = y + static_cast<size_t>(row) * cols;

    float sum_sq = 0.0f;
    for (int col = threadIdx.x; col < cols; col += BLOCK_SIZE) {
        float v = row_in[col];
        sum_sq += v * v;
    }

    sum_sq = block_reduce_sum<BLOCK_SIZE>(sum_sq);

    __shared__ float s_inv_rms;
    if (threadIdx.x == 0) {
        s_inv_rms = rsqrtf(sum_sq / static_cast<float>(cols) + eps);
    }
    __syncthreads();

    float inv_rms = s_inv_rms;
    for (int col = threadIdx.x; col < cols; col += BLOCK_SIZE) {
        row_out[col] = row_in[col] * inv_rms * weight[col];
    }
}
```

### 3.2 Week 3 实战命令行
```bash
# 运行 Rows=4096, HiddenDim in {1024, 4096, 8192, 32768} 性能对比
./build/norm_softmax

# 抓取 Online Softmax 与 RMSNorm 的 NCU 报告（重点看 SFU 利用率与 DRAM 带宽）
ncu --set full --import-source yes --profile-from-start off -f -o reports/b200_norm_softmax ./build/norm_softmax 4096 8192

# 检查 __shfl_xor_sync (SHFL.BFLY) 与 expf (MUFU.EX2) 的硬件指令生成
cuobjdump --dump-sass build/norm_softmax | grep -E "SHFL|MUFU" -C 3
```

---

## Week 4：IO-aware 简化版 FlashAttention 与 KV Cache

### 4.1 核心代码（`src/week04/attention_kv.cu` 融合 Attention 与 Decode KV Cache）
```cuda
#include "cuda_check.h"
#include <cfloat>
#include <cmath>

// ============================================================================
// 1. IO-aware 简化版 FlashAttention Forward (单 Head 视图，支持 Causal Mask)
// 彻底消除 O(N^2) 中间矩阵 S 和 P 的 Global Memory 读写！
// 每个 Block 负责 Q 的 Br 行，外层循环遍历 K, V 的 Bc 列分块
// ============================================================================
template <int Br = 32, int Bc = 32, int D = 64>
__global__ void flash_attention_forward_kernel(
    const float* __restrict__ Q, // [B, H, N, D]
    const float* __restrict__ K, // [B, H, N, D]
    const float* __restrict__ V, // [B, H, N, D]
    float* __restrict__ O,       // [B, H, N, D]
    int N, float scale, bool is_causal) {
    int bh_idx = blockIdx.y; // Batch * Head 索引
    int q_tile = blockIdx.x; // 当前负责的 Q 分块号
    int tx = threadIdx.x;    // 0 .. Br-1 (每个线程负责 Q 的 1 行)

    int q_row = q_tile * Br + tx;
    size_t bh_offset = static_cast<size_t>(bh_idx) * N * D;

    const float* Q_bh = Q + bh_offset;
    const float* K_bh = K + bh_offset;
    const float* V_bh = V + bh_offset;
    float*       O_bh = O + bh_offset;

    __shared__ float s_K[Bc][D];
    __shared__ float s_V[Bc][D];

    // 将当前线程负责的 Q[q_row, 0..D-1] 加载到私有寄存器
    float q_reg[D];
    float o_acc[D] = {0.0f};
    if (q_row < N) {
        for (int d = 0; d < D; ++d) q_reg[d] = Q_bh[q_row * D + d];
    }

    float m_i = -FLT_MAX; // 历史局部最大值
    float l_i = 0.0f;     // 历史局部指数和

    // Causal Mask 优化：对角线上方的 KV 分块直接跳过（节省 50% 计算）
    int max_kv_tiles = is_causal ? min((N + Bc - 1) / Bc, ((q_tile + 1) * Br + Bc - 1) / Bc)
                                 : (N + Bc - 1) / Bc;

    for (int kv_tile = 0; kv_tile < max_kv_tiles; ++kv_tile) {
        int kv_row = kv_tile * Bc + tx;

        // 协作加载 K_j 和 V_j 分块到 Shared Memory
        if (kv_row < N) {
            for (int d = 0; d < D; ++d) {
                s_K[tx][d] = K_bh[kv_row * D + d];
                s_V[tx][d] = V_bh[kv_row * D + d];
            }
        } else {
            for (int d = 0; d < D; ++d) {
                s_K[tx][d] = 0.0f;
                s_V[tx][d] = 0.0f;
            }
        }
        __syncthreads();

        if (q_row < N) {
            float s_ij[Bc];
            float m_ij = -FLT_MAX;

            // 1. 片上计算局部注意力分数 S_ij = Q_i * K_j^T * scale，并求分块最大值 m_ij
            for (int j = 0; j < Bc; ++j) {
                int k_col = kv_tile * Bc + j;
                if (k_col >= N || (is_causal && k_col > q_row)) {
                    s_ij[j] = -FLT_MAX;
                } else {
                    float dot = 0.0f;
                    #pragma unroll
                    for (int d = 0; d < D; ++d) dot += q_reg[d] * s_K[j][d];
                    s_ij[j] = dot * scale;
                    m_ij = fmaxf(m_ij, s_ij[j]);
                }
            }

            // 2. Online Softmax 递推更新 m_new, l_new 并重缩放历史输出累加器 o_acc
            float m_new = fmaxf(m_i, m_ij);
            float alpha = (m_i == -FLT_MAX) ? 0.0f : expf(m_i - m_new);
            float l_ij = 0.0f;

            #pragma unroll
            for (int d = 0; d < D; ++d) o_acc[d] *= alpha;

            for (int j = 0; j < Bc; ++j) {
                float p_ij = (s_ij[j] == -FLT_MAX) ? 0.0f : expf(s_ij[j] - m_new);
                l_ij += p_ij;
                #pragma unroll
                for (int d = 0; d < D; ++d) o_acc[d] += p_ij * s_V[j][d];
            }

            l_i = l_i * alpha + l_ij;
            m_i = m_new;
        }
        __syncthreads();
    }

    // 3. 最终归一化写回 Global Memory O
    if (q_row < N && l_i > 0.0f) {
        float inv_l = 1.0f / l_i;
        for (int d = 0; d < D; ++d) {
            O_bh[q_row * D + d] = o_acc[d] * inv_l;
        }
    }
}

// ============================================================================
// 2. LLM Decode 阶段单 Token Attention (Sq = 1, 读取历史 KV Cache)
// 典型 Memory-Bound 负载：每个 Block 处理一个 (Batch, Head) 对
// ============================================================================
template <int BLOCK_SIZE = 256, int D = 64>
__global__ void decode_kv_cache_attention_kernel(
    const float* __restrict__ q,       // [B, H, 1, D]
    const float* __restrict__ k_cache, // [B, H, MaxSeqLen, D]
    const float* __restrict__ v_cache, // [B, H, MaxSeqLen, D]
    float* __restrict__ out,           // [B, H, 1, D]
    int kv_len, int max_seq_len, float scale) {
    int bh_idx = blockIdx.x;
    int tid = threadIdx.x;

    const float* q_vec = q + static_cast<size_t>(bh_idx) * D;
    const float* k_bh  = k_cache + static_cast<size_t>(bh_idx) * max_seq_len * D;
    const float* v_bh  = v_cache + static_cast<size_t>(bh_idx) * max_seq_len * D;
    float*       o_vec = out + static_cast<size_t>(bh_idx) * D;

    // 预加载当前单 Token Query 到 Shared Memory
    __shared__ float s_q[D];
    if (tid < D) s_q[tid] = q_vec[tid];
    __syncthreads();

    // 每个线程遍历一部分历史 KV Token，维护线程级 Online Softmax 状态
    float m_local = -FLT_MAX;
    float l_local = 0.0f;
    float o_local[D] = {0.0f};

    for (int pos = tid; pos < kv_len; pos += BLOCK_SIZE) {
        float dot = 0.0f;
        #pragma unroll
        for (int d = 0; d < D; ++d) {
            dot += s_q[d] * k_bh[pos * D + d];
        }
        float score = dot * scale;

        float m_new = fmaxf(m_local, score);
        float alpha = (m_local == -FLT_MAX) ? 0.0f : expf(m_local - m_new);
        float p = expf(score - m_new);

        #pragma unroll
        for (int d = 0; d < D; ++d) {
            o_local[d] = o_local[d] * alpha + p * v_bh[pos * D + d];
        }
        l_local = l_local * alpha + p;
        m_local = m_new;
    }

    // 跨线程规约 (m_local, l_local, o_local[D]) 至 Shared Memory
    __shared__ float s_m[BLOCK_SIZE];
    __shared__ float s_l[BLOCK_SIZE];
    s_m[tid] = m_local;
    s_l[tid] = l_local;
    __syncthreads();

    // 由 tid == 0 求全局最大值 m_global 并广播
    __shared__ float s_m_global;
    if (tid == 0) {
        float mg = -FLT_MAX;
        for (int i = 0; i < BLOCK_SIZE; ++i) mg = fmaxf(mg, s_m[i]);
        s_m_global = mg;
    }
    __syncthreads();

    float m_global = s_m_global;
    float scale_thread = (m_local == -FLT_MAX) ? 0.0f : expf(m_local - m_global);
    s_l[tid] = l_local * scale_thread;
    __syncthreads();

    // 将各线程按 m_global 修正后的 o_local 累加到输出
    __shared__ float s_out[D];
    if (tid < D) s_out[tid] = 0.0f;
    __syncthreads();

    for (int d = 0; d < D; ++d) {
        atomicAdd(&s_out[d], o_local[d] * scale_thread);
    }
    __syncthreads();

    if (tid < D) {
        float l_global = 0.0f;
        for (int i = 0; i < BLOCK_SIZE; ++i) l_global += s_l[i];
        o_vec[tid] = s_out[tid] / l_global;
    }
}
```

### 4.2 Week 4 实战命令行
```bash
# 运行 Prefill (N=2048) vs Decode (Sq=1, Skv=2048) 及 Naive vs FlashAttention 对比
./build/attention_kv

# 导出 NCU 报告对比 DRAM 读写流量下降（验证消除 O(N^2) 中间矩阵）与 Roofline 漂移
ncu --set full --import-source yes --profile-from-start off -f -o reports/b200_flash_attn ./build/attention_kv

# 检查融合算子是否因 HeadDim 过大触发寄存器溢出 (LDL / STL)
cuobjdump --dump-sass build/attention_kv | grep -E "LDL|STL" -C 2
```