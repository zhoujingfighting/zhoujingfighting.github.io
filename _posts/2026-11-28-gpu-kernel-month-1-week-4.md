---
title: "GPU Kernel 实战（Month 1 Week 4）：IO-aware 简化版 FlashAttention 与 KV Cache"
date: 2026-11-28
categories: [gpu-kernel]
tags: [GPU Kernel 实战, CUDA, FlashAttention, KV Cache, 算子融合, Blackwell, Nsight Compute]
toc: true
---

> 本文为 **GPU Kernel 实战（Month 1）** 系列第 4 周（Week 4），实现 **IO-aware 简化版 FlashAttention 与 Decode KV Cache**，通过 Nsight Compute 验证消除 $O(N^2)$ 中间矩阵后的 DRAM 流量下降与 Roofline 漂移。
> **硬件目标平台**：NVIDIA B200（Blackwell `sm_100`, 148 SMs, 126.5 MB L2 Cache）｜**工程目录**：`~/workspace/llm-gpu-compute-lab`

```mermaid
mindmap
  root((Month 1 Week 4 核心图谱))
    IO-aware 注意力融合
      简化版 FlashAttention
      Prefill vs Decode KV Cache
      消除 O(N^2) 中间矩阵写回
      在线 Softmax 状态维护
    性能剖析与溢出排查
      DRAM 读写流量对比
      Roofline 模型算力漂移
      HeadDim 增大寄存器压力
      排查 LDL/STL 寄存器溢出
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

---

> 👈 **上一周回顾**：[**Week 3：Softmax 与 LayerNorm / RMSNorm 算子融合**](/gpu-kernel/2026-11-27-gpu-kernel-month-1-week-3/)
> 🏷️ **专题合集**：查看全部实战笔记，请访问 [**GPU Kernel 实战标签页**](/tags/GPU-Kernel-实战/)。
