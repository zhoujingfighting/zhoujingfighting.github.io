---
title: "GPGPU 编译器面试题精选（Day 27）：VOP3P 双发射解码、TokenFactor 压缩、Fallthrough 对齐与 Tile 自动调优"
date: 2026-11-11
categories: [gpu-compiler]
tags: [GPGPU, LLVM, MLIR, Triton, 指令发射, 指令调度, Auto-Tuning, 面试]
toc: true
---

> 本文为 **[GPGPU 编译器工程师面试通关指南](/gpgpu-interview/)** 系列第27期（Day 27）。
> 建议在阅读“面试官参考解析”前，先独立思考每道题目的考察意图与答题脉络（每题思考 3~5 分钟）。

> 💡 **核心考察维度**：AMDGPUDisassembler VOP3P 扩展打包与双发射 (Dual-Issue) 机器字解码、SelectionDAG TokenFactor 展开与内存节点扇入 (Fan-In) 树状压缩、MachineBlockPlacement 分支概率感知块对齐与 Fallthrough 路径最大化、ConvertGPUToLLVMPass Subgroup All/Any 投票原语映射与 Active Mask 规约、Triton/AI 编译器寄存器脚印驱动的分块调优 (Tile Tuning) 与边界锯齿消除。

---

## 📝 今日高频面试题清单

### 【Q1 · 双发射与反汇编解码】AMDGPUDisassembly.cpp 与 MCDisassembler 中的 VOP3P 扩展打包与双发射（Dual-Issue）解码
在现代 AMDGPU 微架构（如 CDNA2/3 或 RDNA3）中，硬件引入了 VOP3P 编码格式以支持双 16 位打包算术（Packed Math，如 V_PK_FMA_F16）以及波前双发射（Dual-Issue Wavefront Execution）。请结合 llvm/lib/Target/AMDGPU/Disassembler/AMDGPUDisassembler.cpp 源码，分析 MCDisassembler 是如何解码 VOP3P 指令中复杂的操作数修饰符（如 NegHi/NegLo、AbsHi/AbsLo）的？从后端指令发射的角度看，指令调度器是如何识别相邻指令满足 dual-issue 规则并在编码时设置特化发射位的？

---

### 【Q2 · 依赖压缩与链条展开】SelectionDAG/ScheduleDAGSDNodes.cpp 中的 TokenFactor 展开与内存节点扇入（Fan-In）压缩
在复杂算子的大规模展开中，基本块内可能并发存在数十条异步内存加载与存储。若每条指令都保留独立的依赖链，会导致内存控制依赖图（Chain Graph）的边数急剧膨胀，拖垮调度器。请结合 llvm/lib/CodeGen/SelectionDAG/SelectionDAG.cpp 与 ScheduleDAGSDNodes.cpp 源码，分析 `ISD::TokenFactor` 节点是如何将多个平行的内存前驱压缩合并为单个令牌因子的？在调度器构建 SUnit 依赖图时，`ScheduleDAGSDNodes` 如何避免因深层嵌套的 TokenFactor 造成二次方（$O(N^2)$）的冗余依赖边扩散？

---

### 【Q3 · 分支预测与基本块对齐】MachineBlockPlacement.cpp 中的分支概率感知块对齐与 Fallthrough 路径最大化
在 GPU 机器指令发射中，无条件绝对跳转（Unconditional Branch）不仅浪费指令发射槽，还会在指令缓存（I-Cache）中产生气泡。请结合 llvm/lib/CodeGen/MachineBlockPlacement.cpp 源码，分析 MachineBlockPlacement 是如何基于分支概率分析（BranchProbabilityInfo）构建块链（Block Chains）的？在面对 SIMT 的发散分支时，编译器为什么会极度倾向于将最可能收敛执行的热路径排布为直通块（Fallthrough），并将罕见冷路径（如异常/捕获块）剥离到物理内存远处？

---

### 【Q4 · 子图投票与掩码规约】ConvertGPUToLLVMPass.cpp 中的 Subgroup All/Any 投票原语映射与 Active Mask 规约
在跨线程条件检测中，OpenCL / Vulkan / CUDA 提供了投票原语（如 `gpu.all` 与 `gpu.any`）。请结合 mlir/lib/Conversion/GPUCommon/GPUOpsLowering.cpp 源码，分析 `GPUAllReduceOpLowering` 或 `GPUSubgroupReduceOpLowering` 是如何根据目标平台将其映射为底层投票指令的？在 NVPTX 降级为 `@llvm.nvvm.vote.all.sync` 与在 ROCDL 降级为 `v_cmpx` 或 `s_and_b32` 时，硬件在微架构层面的执行机制有何本质差异？

---

### 【Q5 · 寄存器驱动分块调优】Triton / AI 编译器中的 Register Footprint Driven Tile Tuning 与边界锯齿消除
在 AI 编译器生成 GEMM 算子时，Tile 尺寸的选取直接决定了寄存器足迹（Register Footprint）。例如选择 $128 	imes 128 	imes 32$ 的分块，单个线程持有的累加器、A 矩阵切片和 B 矩阵切片需要的 VGPR 数量可能恰好处于占用率阶梯的突变点（如突破 128 个寄存器导致单 SM 驻留 Block 数从 2 掉落至 1）。请结合 AI 编译器架构分析：静态调优器（Autotuner / Cost Model）如何通过代数建模精确预估各个 Tile 维度的寄存器用量？当张量真实尺寸与最优分块不匹配时，编译器如何通过多维虚拟切片（Virtual Slicing）消除边界锯齿（Edge Sawtooth）对内存吞吐的削弱？

---

## 🎯 面试官深度解析与追问点

```mermaid
mindmap
  root((Day 27 核心考点图谱))
    VOP3P 双发射解码
      AMDGPUDisassembler
      packed 16-bit 向量指令
      双发射 (Dual-Issue) 指令槽位
      主从操作数打包与解码
    TokenFactor 依赖压缩
      SelectionDAG
      内存屏障与链节点 (Chains)
      超大扇入 (Fan-In) 树状重构
      拓扑排序与内存占用剪枝
    分支概率感知块对齐
      MachineBlockPlacement
      热路径 Fallthrough 最大化
      消除无谓的无条件跳转
      指令预取命中率保护
    Subgroup 投票与规约
      ConvertGPUToLLVMPass
      gpu.subgroup_reduce (all/any)
      PTX vote.sync 原语
      Active Mask 掩码收敛
    寄存器驱动 Tile 调优
      Triton Auto-Tuning
      编译期寄存器脚印 (Footprint) 模拟
      边界锯齿 (Tail Quantization) 消除
      Occupancy 与 Tile 尺寸全局寻优
```

---

### 题 1 深度解析：AMDGPUDisassembly.cpp 与 MCDisassembler 中的 VOP3P 扩展打包与双发射（Dual-Issue）解码

#### 源码级深度剖析与核心实现
> 📌 **源码剖析提问**：在现代 AMDGPU 微架构（如 CDNA2/3 或 RDNA3）中，硬件引入了 VOP3P 编码格式以支持双 16 位打包算术（Packed Math，如 V_PK_FMA_F16）以及波前双发射（Dual-Issue Wavefront Execution）。请结合 llvm/lib/Target/AMDGPU/Disassembler/AMDGPUDisassembler.cpp 源码，分析 MCDisassembler 是如何解码 VOP3P 指令中复杂的操作数修饰符（如 NegHi/NegLo、AbsHi/AbsLo）的？从后端指令发射的角度看，指令调度器是如何识别相邻指令满足 dual-issue 规则并在编码时设置特化发射位的？

##### 源码文件直达

  - `llvm/lib/Target/AMDGPU/Disassembler/AMDGPUDisassembler.cpp`
  - `llvm/lib/Target/AMDGPU/MCTargetDesc/SIMCCodeEmitter.cpp`

##### 核心类与核心代码块逻辑

  在 `AMDGPUDisassembler.cpp` 中，解码器解析 VOP3P 独特的 64 位指令字与修饰符位段：

```cpp
// llvm/lib/Target/AMDGPU/Disassembler/AMDGPUDisassembler.cpp 核心逻辑精选
DecodeStatus AMDGPUDisassembler::decodeVOP3POperands(
    MCInst &MI, uint64_t Val, uint64_t Address, const MCDisassembler *Decoder) {
  // 1. 提取输入源操作数与目标操作数字段
  unsigned Src0 = fieldFromInstruction(Val, 0, 9);
  unsigned Src1 = fieldFromInstruction(Val, 9, 9);
  unsigned Src2 = fieldFromInstruction(Val, 18, 9);
  unsigned Dst  = fieldFromInstruction(Val, 56, 8);

  // 2. 解码 Packed 专用修饰符掩码 (Neg_Lo: 47, Neg_Hi: 48, Opsel_Hi: 59 等)
  unsigned NegLo = fieldFromInstruction(Val, 45, 3);
  unsigned NegHi = fieldFromInstruction(Val, 48, 3);
  unsigned OpselHi = fieldFromInstruction(Val, 59, 3);

  // 3. 构建 MCInst 操作数并打上位段修饰符属性
  MI.addOperand(MCOperand::createReg(getVGPR(Dst)));
  MI.addOperand(MCOperand::createImm(NegLo));
  MI.addOperand(MCOperand::createImm(NegHi));
  MI.addOperand(MCOperand::createImm(OpselHi));
  return MCDisassembler::Success;
}
```

##### 逐行机械执行分析

  1. **双通道修饰符硬件解耦（Packed Modifiers）**：VOP3P 格式允许单条 64 位指令同时对高 16 位和低 16 位施加完全独立的算术变换。`NegLo` 和 `NegHi` 分别控制两个独立 ALU 通道的取反电路，`OpselHi` 则控制源操作数是选择原 32 位寄存器的高半字还是低半字。
  2. **双发射（Dual-Issue）微架构仲裁**：在 RDNA3 等架构中，单个 Compute Unit 包含双发射算术通路。后端的 `SIInsertHardClauses` 或 `PostRASchedulerList` 必须确保相邻的两条指令满额满足：
     - 指令 A 与指令 B 无 RAW/WAR 物理寄存器冲突；
     - 指令 A 和指令 B 不能同时请求同一套受限资源（如只能有 1 条是全局访存或 Transcendental 特殊函数）；
  3. **编码字段对齐**：当检测到满足双发射条件时，MCCodeEmitter 会在第一条指令的特定控制位打上 `DUAL_ISSUE_BIT`，硬件发射控制器在单时钟周期将两束微码并行打入主执行管道。

---

### 题 2 深度解析：SelectionDAG/ScheduleDAGSDNodes.cpp 中的 TokenFactor 展开与内存节点扇入（Fan-In）压缩

#### 源码级深度剖析与核心实现
> 📌 **源码剖析提问**：在复杂算子的大规模展开中，基本块内可能并发存在数十条异步内存加载与存储。若每条指令都保留独立的依赖链，会导致内存控制依赖图（Chain Graph）的边数急剧膨胀，拖垮调度器。请结合 llvm/lib/CodeGen/SelectionDAG/SelectionDAG.cpp 与 ScheduleDAGSDNodes.cpp 源码，分析 `ISD::TokenFactor` 节点是如何将多个平行的内存前驱压缩合并为单个令牌因子的？在调度器构建 SUnit 依赖图时，`ScheduleDAGSDNodes` 如何避免因深层嵌套的 TokenFactor 造成二次方（$O(N^2)$）的冗余依赖边扩散？

##### 源码文件直达

  - `llvm/lib/CodeGen/SelectionDAG/SelectionDAG.cpp`
  - `llvm/lib/CodeGen/SelectionDAG/ScheduleDAGSDNodes.cpp`

##### 核心类与核心代码块逻辑

  在 `SelectionDAG.cpp` 中，`getNode(ISD::TokenFactor)` 动态消除嵌套链条并扁平化操作数列表：

```cpp
// llvm/lib/CodeGen/SelectionDAG/SelectionDAG.cpp 核心逻辑精选
SDValue SelectionDAG::getTokenFactor(const SDLoc &DL,
                                     SmallVectorImpl<SDValue> &Ops) {
  // 1. 过滤掉 EntryToken 与冗余节点
  SmallVector<SDValue, 8> FlatOps;
  for (SDValue Op : Ops) {
    if (Op.getOpcode() == ISD::EntryToken) continue;
    // 2. 若子操作数本身也是 TokenFactor，将其操作数解压扁平化展平
    if (Op.getOpcode() == ISD::TokenFactor) {
      for (const SDValue &SubOp : Op.getNode()->op_values())
        FlatOps.push_back(SubOp);
    } else {
      FlatOps.push_back(Op);
    }
  }

  // 3. 排序去重，避免重复依赖污染
  llvm::sort(FlatOps);
  FlatOps.erase(std::unique(FlatOps.begin(), FlatOps.end()), FlatOps.end());

  if (FlatOps.empty()) return getEntryNode();
  if (FlatOps.size() == 1) return FlatOps[0];

  return getNode(ISD::TokenFactor, DL, MVT::Other, FlatOps);
}
```

##### 逐行机械执行分析

  1. **多依赖扇入压缩（Fan-In Collapse）**：当一个基本块发生条件汇聚或执行同步屏障时，可能需要等待上百条独立的全局 Load 完成。若直接为屏障节点连入 100 条前驱边，图搜索开销剧烈膨胀。`TokenFactor` 扮演了逻辑汇流器的角色，将散落的并发令牌收拢为单一的 `MVT::Other` 依赖节点。
  2. **递归扁平化展开**：若通过多次局部合并产生了树状嵌套的 `TokenFactor(TokenFactor(A, B), C)`，`SelectionDAG::getTokenFactor` 递归将内部节点剥离为扁平单层向量，并在原地调用 `std::unique` 彻底抹除重复引入的同一祖先节点。
  3. **拓扑传递闭包阻断（Transit Elimination）**：在 `ScheduleDAGSDNodes::BuildSchedUnits` 中，调度器遍历 `TokenFactor` 时并不会盲目将所有叶子节点的依赖全量广播给后继节点，而是仅挂接边界节点，有效遏制了调度图的边数平方爆炸，保证了上千条指令级别的调度器收敛速度。

---

### 题 3 深度解析：MachineBlockPlacement.cpp 中的分支概率感知块对齐与 Fallthrough 路径最大化

#### 源码级深度剖析与核心实现
> 📌 **源码剖析提问**：在 GPU 机器指令发射中，无条件绝对跳转（Unconditional Branch）不仅浪费指令发射槽，还会在指令缓存（I-Cache）中产生气泡。请结合 llvm/lib/CodeGen/MachineBlockPlacement.cpp 源码，分析 MachineBlockPlacement 是如何基于分支概率分析（BranchProbabilityInfo）构建块链（Block Chains）的？在面对 SIMT 的发散分支时，编译器为什么会极度倾向于将最可能收敛执行的热路径排布为直通块（Fallthrough），并将罕见冷路径（如异常/捕获块）剥离到物理内存远处？

##### 源码文件直达

  - `llvm/lib/CodeGen/MachineBlockPlacement.cpp`
  - `llvm/lib/Analysis/BranchProbabilityInfo.cpp`

##### 核心类与核心代码块逻辑

  在 `MachineBlockPlacement.cpp` 中，布局算法贪心寻找转移概率最大的后继块构成链：

```cpp
// llvm/lib/CodeGen/MachineBlockPlacement.cpp 核心逻辑精选
MachineBasicBlock *MachineBlockPlacement::selectBestSuccessor(
    const MachineBasicBlock *BB, const BlockChain &Chain,
    BranchProbability &BestProb) {
  MachineBasicBlock *BestSucc = nullptr;

  // 遍历当前基本块的所有控制流后继
  for (MachineBasicBlock *Succ : BB->successors()) {
    // 1. 获取边缘转移概率 (Branch Probability)
    BranchProbability Prob = MBPI->getEdgeProbability(BB, Succ);

    // 2. 检查后继块是否已经在其他链中锁定，并且不是当前热链的前驱
    if (isLoopHeader(Succ) || !canBeFallthrough(BB, Succ))
      continue;

    // 3. 贪心选取概率突破阈值的最优直通目标
    if (Prob > BestProb) {
      BestProb = Prob;
      BestSucc = Succ;
    }
  }
  return BestSucc;
}
```

##### 逐行机械执行分析

  1. **消除无条件跳步（Branch Elimination）**：若基本块 $A$ 执行完毕后跳向 $B$，且 $B$ 在物理内存中紧跟在 $A$ 之后，汇编器可以直接抹去 $A$ 尾部的无条件跳转指令 `s_branch B`，指令流直接跨越物理边界进入 $B$，指令发射槽消耗为零。
  2. **SIMT 发散惩罚隔离**：在 GPU 场景下，某些条件分支属于极低概率事件（例如处理矩阵乘边界越界时的保护分支，或动态除零异常校验）。`selectBestSuccessor` 依据静态 Profile 或静态启发式（冷分支概率低于 1%），将越界保护块剥离出主干。
  3. **I-Cache 行预取收益**：主流计算核心以 64/128 字节为颗粒度从指令 Cache 中预取指令。通过将热路径 100% 紧凑排列在连续内存地址中，Warp 执行时流水线始终处于自然连续读取状态，避免了由于执行远端跳转造成的指令缓存行颠簸（Cache Line Thrashing）。

---

### 题 4 深度解析：ConvertGPUToLLVMPass.cpp 中的 Subgroup All/Any 投票原语映射与 Active Mask 规约

#### 源码级深度剖析与核心实现
> 📌 **源码剖析提问**：在跨线程条件检测中，OpenCL / Vulkan / CUDA 提供了投票原语（如 `gpu.all` 与 `gpu.any`）。请结合 mlir/lib/Conversion/GPUCommon/GPUOpsLowering.cpp 源码，分析 `GPUAllReduceOpLowering` 或 `GPUSubgroupReduceOpLowering` 是如何根据目标平台将其映射为底层投票指令的？在 NVPTX 降级为 `@llvm.nvvm.vote.all.sync` 与在 ROCDL 降级为 `v_cmpx` 或 `s_and_b32` 时，硬件在微架构层面的执行机制有何本质差异？

##### 源码文件直达

  - `mlir/lib/Conversion/GPUCommon/GPUOpsLowering.cpp`
  - `llvm/lib/Target/NVPTX/NVPTXISelLowering.cpp`

##### 核心类与核心代码块逻辑

  在模式降级中，布尔投票操作被重写为专有的硬件硬件汇总指令：

```cpp
// mlir/lib/Conversion/GPUCommon/GPUOpsLowering.cpp 核心逻辑精选
LogicalResult GPUSubgroupVoteOpLowering::matchAndRewrite(
    gpu::SubgroupVoteOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  Value predicate = adaptor.getValue();

  if (isNVVM) {
    // 1. NVPTX 模式: 传入全掩码 (0xffffffff) 并发射投票 Intrinsic
    Value mask = rewriter.create<arith::ConstantIntOp>(loc, 0xffffffff, 32);
    if (op.getOp() == gpu::AllOp::ALL) {
      rewriter.replaceOpWithNewOp<NVVM::VoteAllSyncOp>(op, rewriter.getI1Type(), mask, predicate);
    } else {
      rewriter.replaceOpWithNewOp<NVVM::VoteAnySyncOp>(op, rewriter.getI1Type(), mask, predicate);
    }
    return success();
  } else if (isROCDL) {
    // 2. ROCDL 模式: 发射硬件比较指令，将结果写入标量寄存器 (VCC)
    // 利用波前标量比较: v_cmp_ne_u32 vcc, predicate, 0
    // 然后利用 s_and_b32 / s_or_b32 对 VCC 与 EXEC 进行规约并检查 SCC 状态位
    lowerVoteToROCDL(rewriter, loc, op, predicate);
    return success();
  }
  return failure();
}
```

##### 逐行机械执行分析

  1. **NVIDIA 微架构硬件投票单元（Dedicated Vote Unit）**：在 Volta 及之后架构中，SM 内部配备了专职的 Warp 投票硬件单元。`VoteAllSync` 指令发射后，32 个 Lane 的 1 位谓词在单时钟周期内被并行推入硬件总线，由硬连线树形门电路直接产出最终的布尔汇总结果，并回传给每个线程的私有寄存器，无需借道标量流水线。
  2. **AMD 基于标量执行单元（SALU）的 VCC/EXEC 规约**：AMD GPU 架构没有独立的硬件投票单元，而是直接借用标量流水线。
     - 向量单元发射 `v_cmp_ne_u32 vcc, pred, 0`，将各个 Lane 的布尔状态生成为一个 64 位（Wave64）或 32 位（Wave32）的位掩码写入 `VCC`；
     - 标量单元发射 `s_and_b64 s[0:1], vcc, exec`。若结果严格等于 `exec`，则判定全真（All 为真）；若结果非零，则判定存在真（Any 为真）。这一过程直接复用了标量 ALU 的状态位（`SCC`），开销极小。

---

### 题 5 深度解析：Triton / AI 编译器中的 Register Footprint Driven Tile Tuning 与边界锯齿消除

#### 源码级深度剖析与核心实现
> 📌 **源码剖析提问**：在 AI 编译器生成 GEMM 算子时，Tile 尺寸的选取直接决定了寄存器足迹（Register Footprint）。例如选择 $128 	imes 128 	imes 32$ 的分块，单个线程持有的累加器、A 矩阵切片和 B 矩阵切片需要的 VGPR 数量可能恰好处于占用率阶梯的突变点（如突破 128 个寄存器导致单 SM 驻留 Block 数从 2 掉落至 1）。请结合 AI 编译器架构分析：静态调优器（Autotuner / Cost Model）如何通过代数建模精确预估各个 Tile 维度的寄存器用量？当张量真实尺寸与最优分块不匹配时，编译器如何通过多维虚拟切片（Virtual Slicing）消除边界锯齿（Edge Sawtooth）对内存吞吐的削弱？

##### 核心算法模型与自动调优器直达

  - 在 Triton 及 TVM TensorIR 中，寄存器预测器与分块重映射算法驱动内核参数选择。

##### 寄存器足迹代数预测器伪代码模型

```cpp
// 典型的 AI 编译器寄存器足迹预测代数模型
struct RegisterFootprint {
  unsigned AccVGPRs;
  unsigned SmemLoadVGPRs;
  unsigned AddressVGPRs;

  unsigned getTotalVGPRs() const {
    return AccVGPRs + SmemLoadVGPRs + AddressVGPRs + 8 /* 预留编译器临时寄存器 */;
  }
};

RegisterFootprint estimateTileRegisters(int BLOCK_M, int BLOCK_N, int BLOCK_K,
                                       int numWarps, Type elemType) {
  RegisterFootprint FP;
  int threadsTotal = numWarps * 32;

  // 1. 矩阵乘累加器寄存器 (通常为 FP32 累加): (BLOCK_M * BLOCK_N) / threadsTotal
  FP.AccVGPRs = (BLOCK_M * BLOCK_N) / threadsTotal;

  // 2. 双缓冲片上加载寄存器 (A Fragment + B Fragment):
  int numElementsA = (BLOCK_M * BLOCK_K) / threadsTotal;
  int numElementsB = (BLOCK_K * BLOCK_N) / threadsTotal;
  // FP16 两个元素打包占 1 个 32-bit VGPR
  FP.SmemLoadVGPRs = (numElementsA + numElementsB) / 2;

  // 3. 全局基址及步长指针所需寄存器
  FP.AddressVGPRs = 6;
  return FP;
}
```

##### 逐行机械执行分析

  1. **累加器占据绝对主导**：在计算密集算子中，累加器必须全程驻留在寄存器中，无法溢出或中途释放。对于 $BLOCK\_M=128, BLOCK\_N=128, numWarps=4 (128 threads)$，仅累加器就需要：
     $$	ext{AccVGPRs} = 
rac{128 	imes 128}{128} = 128 	ext{ 个 32-bit VGPR}$$
     此时单线程已占满 128 寄存器。若算上地址变量与载入缓冲，总寄存器数必然突破 128（达到 140+），硬件 Occupancy 瞬间腰斩。调优器通过上述公式在编译前静态剪枝，强制将 $BLOCK\_N$ 调降至 64 或提升 `numWarps` 至 8，将总寄存器压回 128 阈值内，换取翻倍的驻留 Block 数。
  2. **边界锯齿消除（Tail Slicing / Wavefront Remapping）**：当输入张量维度为不规整形状（如 $M=1000$，无法被 $BLOCK\_M=128$ 整除，最后产生长为 104 的残缺块），若直接走慢速的 Padding 循环，硬件 LSU 的 128 字节合并访存会被破坏。编译器利用 **Virtual Slicing** 将残缺块重新分解为规整的 $64 	imes 64$ 或 $32 	imes 32$ 子网格，交由独立的微内核（Peeled Micro-kernel）全速发射，主干网格始终以 100% 满带宽运行，消除边缘锯齿带来的总线饥饿。

---

## 5 深度技术对比

| 优化阶段 / 机制 | 解决的核心硬件/软件瓶颈 | LLVM/MLIR/Triton 关键接口与节点 | 错误处理不当的严重后果 |
| :---- | :---- | :---- | :---- |
| **VOP3P Dual-Issue** | 消除传统单标量发射对 16 位浮点/整型并行计算密度的压制 | `VOP3P`, `AMDGPUDisassembler` | 错失硬件双倍算力吞吐，发射流水线出现空等气泡 |
| **TokenFactor Compression** | 解决大量内存读写依赖汇聚导致 SelectionDAG 节点数指数爆炸 | `ISD::TokenFactor`, `SelectionDAG` | 编译期内存耗尽 OOM，或调度器陷入死循环拓扑遍历 |
| **Block Placement Alignment** | 消除高频热路径分支由于跨 Cache Line 对齐不良引发的跳转延迟 | `MachineBlockPlacement`, `BranchProb` | 指令预取带宽被分支气泡吞噬，长内核执行延迟激增 |
| **Subgroup Vote Lowering** | 将抽象的全组条件一致性判定降级为硬件单周期位掩码原语 | `vote.sync`, `v_readlane` | 逻辑判定演化为昂贵的线程间共享内存同步，延迟剧增 |
| **Register-Driven Tuning** | 解决盲目增大分块导致寄存器溢出（Spill）带来的性能断崖下跌 | `AutoTuning`, `OccupancyCostModel` | 触发剧烈的显存 Spill 惩罚，算子吞吐暴跌一个数量级 |

---

## 💡 面试实战总结（复盘金句）

> 1. **谈 VOP3P 指令打包与双发射解码时**：*"双发射是挖掘超标量极限潜能的核心武器；把成对的低精度操作在机器字层面熔合为一个原子 VOP3P 单元，是让 ALU 计算管道瞬间迸发双倍火力的精妙技巧。"*
> 2. **谈 TokenFactor 依赖压缩与 DAG 剪枝时**：*"内存链条的无限膨胀是指令选择器的梦魇；通过扁平化展开与冗余剪枝把散乱的 TokenFactor 收敛为轻盈的依赖骨架，是保障极端内核成功编译的关键基础设施。"*
> 3. **谈分支概率感知与基本块直通对齐时**：*"最快的跳转是无需跳转（Fallthrough）；依据静态剖析与概率图景将热路径压实为单向直线流，是编译器对微架构指令预取与分支预测器的最高敬意。"*
> 4. **谈 Subgroup 投票原语与掩码规约时**：*"协同判定是分布式 SIMT 系统的神经突触；把跨线程的布尔决议提炼为单条硬件掩码操作，是以极简指令达成全局共识的优雅范例。"*
> 5. **谈寄存器驱动的分块调优与锯齿消除时**：*"分块几何的黄金分割点永远受制于片上资源的红线；以严格的寄存器物理脚印驱动 Tile 参数搜索，才能在算力饱和与零溢出之间觅得 Pareto 最优解。"*

---

> 👈 **上一期回顾**：[**Day 26：Wave32/64 模式切换、宽整数进位链、Literal 字仲裁与 Warp-Specialized**](/gpu-compiler/2026-11-10-gpgpu-compiler-interview-day-26/)
> 👉 **下一期推荐**：[**Day 28：LSR 寻址代价、TwoAddress 破坏性重写、宽向量 Memcpy 与持久化内核**](/gpu-compiler/2026-11-12-gpgpu-compiler-interview-day-28/)
> 👉 **专栏返回**：查看更多模块内容与更新计划，请访问 [**GPGPU 编译器面试通关指南目录**](/gpgpu-interview/)。