# **GPGPU 编译器模拟面试与深度解析 —— Day 04**

# **1\. GlobalISel 与 SelectionDAG 在 GPU 后端的设计差异**

## **题目描述**

1. 请深入对比 GlobalISel 与 SelectionDAG 在指令选择阶段的核心架构差异。请结合现代 GPU 架构的特点，详细说明：为什么复杂的 GPU 后端（如 AMDGPU、NVPTX）正在全面转向 GlobalISel？传统 SelectionDAG 遇到了哪些无法克服的瓶颈？GlobalISel 的 `LegalizerInfo` 如何利用声明式规则处理窄类型与非常规向量？

## **源码级深度剖析**

* **源码文件直达**：  
  * `llvm/lib/CodeGen/GlobalISel/Legalizer.cpp`  
  * `llvm/include/llvm/CodeGen/GlobalISel/LegalizerInfo.h`  
  * `llvm/lib/Target/AMDGPU/AMDGPULegalizerInfo.cpp`  
* **核心类与核心代码块逻辑**：

在 `LegalizerInfo.cpp` 中，目标架构通过声明式 Fluent API 配置泛型指令（G\_LOAD, G\_ADD 等）的合法化动作：

```c
// llvm/lib/Target/AMDGPU/AMDGPULegalizerInfo.cpp 核心逻辑精选
AMDGPULegalizerInfo::AMDGPULegalizerInfo(const GCNSubtarget &ST) {
  // 定义合法操作数类型：标量 s32/s64，向量 v2s32/v4s32
  getActionDefinitionsBuilder(G_ADD)
    .legalFor({s32, s64})
    .clampScalar(0, s32, s64)     // 窄类型如 i1/i8/i16 向上拓宽至 s32 (WidenScalar)
    .widenScalarToNextPow2(0);

  // 针对 GPU 常见但硬件缺乏原生指令的非常规向量 (如 v3s32 顶点/坐标数据)
  getActionDefinitionsBuilder(G_LOAD)
    .legalForTypesWithMemDesc({{s32, p1, s32, 4}, {v4s32, p1, v4s32, 16}})
    .fewerElementsVector(0, s32); // 拆解为多个独立标量 Load
}
```

* **逐行逻辑深度拆解**：  
  * **克服 DAG 基本块内存膨胀**：SelectionDAG 在基本块内将所有 IR 节点展开为巨大的 DAG 拓扑图，遇到 GPU 深度展开的大循环（如展开后上万条指令的 GEMM 内核）会导致内存消耗和拓扑排序时间呈二次方暴增。GlobalISel 则完全建立在基于 MIR 的全局 SSA 虚拟寄存器图上，将流水线清晰拆分为 `IRTranslator` \-\> `Legalizer` \-\> `RegBankSelect` \-\> `InstructionSelect`，内存线性可控。  
  * **窄类型与非常规向量合法化**：通过 `clampScalar` 和 `fewerElementsVector`，在指令选择前将 GPU 原生不支持的非 2 幂次向量（如 `v3s32`）优雅拆解为合法标量或 2 幂次切片，彻底避免了 SelectionDAG 阶段复杂的类型合法化（Type Legalization）阶段序冲突（Phase Ordering Issues）。

# **2\. Tensor Core / Matrix Core 的 Fragment 寄存器布局与约束**

## **题目描述**

Tensor Core (NVIDIA) 和 Matrix Core (AMD) 要求数据在 Warp/Wavefront 内部线程间进行特定模式的离散分布（Fragment）。编译器在 LLVM IR 阶段如何表征这类高维 Fragment 操作？在后端编译过程中，采取了哪些关键技术防止贪心寄存器分配器（Greedy RA）破坏硬件所需的特定寄存器对齐与线程间物理绑定关系？

## **源码级深度剖析**

2. **源码文件直达**：  
   1. `llvm/lib/Target/NVPTX/NVPTXISelDAGToDAG.cpp`  
   2. `llvm/lib/Target/AMDGPU/SIISelLowering.cpp`  
   3. `llvm/lib/CodeGen/TargetRegisterInfo.cpp`  
3. **核心类与核心代码块逻辑**：

在指令选择阶段，后端通过定义复合超寄存器（Super-Register Class）将 MMA 操作数强行绑定为连续原子元组：

```c
// llvm/lib/Target/NVPTX/NVPTXISelDAGToDAG.cpp 核心逻辑精选
bool NVPTXDAGToDAGISel::SelectMMA(SDNode *N) {
  // 1. 匹配 wmma.mma 节点，获取矩阵累加器及源矩阵切片
  SDLoc DL(N);
  SmallVector<SDValue, 8> Ops;
  
  // 2. 将输入的 Fragment 寄存器组绑定到连续 RegisterTuples
  // 例如对于 16x16x16 FP16 MMA，矩阵 C/D 需 4 个 32-bit 寄存器 (每个存 2 个 FP16)
  SDValue RegTuple = CurDAG->getTargetNode(
      TargetOpcode::REG_SEQUENCE, DL, MVT::Untyped,
      Ops[0], CurDAG->getTargetConstant(NVPTX::sub0, DL, MVT::i32),
      Ops[1], CurDAG->getTargetConstant(NVPTX::sub1, DL, MVT::i32),
      Ops[2], CurDAG->getTargetConstant(NVPTX::sub2, DL, MVT::i32),
      Ops[3], CurDAG->getTargetConstant(NVPTX::sub3, DL, MVT::i32));

  // 3. 发射带严格物理元组约束的机器指令节点
  ReplaceNode(N, CurDAG->getMachineNode(NVPTX::WMMA_MMA_f16_f16, DL, MVT::Untyped, RegTuple));
  return true;
}
```

4. **逐行逻辑深度拆解**：  
   1. `REG_SEQUENCE` **建立物理元组防拆解**：编译器使用 `REG_SEQUENCE` 伪指令将分散的虚拟寄存器打包成单一的超寄存器（如包含 4 个 32 位寄存器的元组）。贪心分配器在着色时必须为其寻找满足 4 寄存器对齐的物理槽位，禁止将不同分量孤立分配到离散寄存器。  
   2. **Late Pseudo Expansion（后置伪指令展开）**：在整个寄存器分配完成之前，MMA 指令始终保持为抽象的伪指令，直到 Post-RA 阶段由 `ExpandPostRAPseudos` 展开为目标机器 ISA，确保线程间的数据切片布局在通过复杂的机器优化 Pass 时不被意外破坏。

# **3\. 发散循环展开（Divergent Loop Unrolling）与谓词化**

## **题目描述**

1. 在 GPU 编程中，发散循环是性能优化的难点。循环展开在面对发散边界和内部跳转时如何权衡完全展开与部分展开？如何通过谓词化与掩码重构控制流？

## **源码级深度剖析**

2. **源码文件直达**：  
   1. `llvm/lib/Transforms/Scalar/LoopUnrollPass.cpp`  
   2. `llvm/include/llvm/Transforms/Utils/UnrollLoop.h`  
3. **核心类与核心代码块逻辑**：

在 `LoopUnrollPass.cpp` 中，针对 GPU 架构对循环展开的 Code Size 和循环展开代价施加保守阈值：

```c
// llvm/lib/Transforms/Scalar/LoopUnrollPass.cpp 核心逻辑精选
bool llvm::computeUnrollCost(const Loop *L, unsigned &Threshold,
                             TargetTransformInfo &TTI) {
  // 1. 若循环终止条件依赖发散变量 (Divergent Trip Count)
  if (TTI.hasBranchDivergence(L->getHeader()->getParent()) &&
      !isUniformTripCount(L)) {
    // 强制调低展开阈值，防止展开后引入大量带有动态谓词的控制流
    Threshold = std::min(Threshold, TTI.getUnrollingThreshold() / 4);
  }

  // 2. 累加循环内部指令代价
  unsigned Cost = 0;
  for (BasicBlock *BB : L->blocks()) {
    for (Instruction &I : *BB) {
      Cost += TTI.getInstructionCost(&I, TargetTransformInfo::TCK_CodeSize);
      if (Cost > Threshold)
        return false; // 超出阈值，禁止展开以防 I-Cache Thrashing
    }
  }
  return true;
}
```

4. **逐行逻辑深度拆解**：  
   1. **发散边界展开惩罚**：若循环的 Trip Count 是发散的（即不同 Lane 循环次数不同），展开该循环必须为每次展开的迭代生成额外的谓词判断（`setp` 指令），导致代码体积急剧膨胀。因此编译器检测到发散边界时会极大压缩展开阈值。  
   2. **全员快速退出重构（Ballot Voting）**：当循环体内包含发散 break 时，编译器会插入 `@llvm.nvvm.vote.ballot.sync` 指令。当且仅当全 Warp 32 个线程的条件掩码全为 0 时，才生成跨越展开体的标量统一跳转，实现高效的 Fast-Path 提前收敛退出。

# **4\. 异步拷贝流水线（TMA / Async Group）与依赖调度**

## **题目描述**

1. 现代 GPU 引入了硬件级异步数据传输机制（如 NVIDIA Ampere/Hopper 的 cp.async / TMA）。指令调度器在处理异步拷贝提交与等待令牌时，必须遵守哪些特殊的依赖约束？若调度错误会引发什么底层后果？

## **源码级深度剖析**

2. **源码文件直达**：  
   1. `llvm/lib/CodeGen/MachineScheduler.cpp`  
   2. `llvm/include/llvm/CodeGen/ScheduleDAG.h`  
3. **核心类与核心代码块逻辑**：

调度器通过在 ScheduleDAG 中为异步令牌与提交组插入不可跨越的屏障依赖边：

```c
// llvm/lib/CodeGen/MachineScheduler.cpp 核心逻辑精选
void ScheduleDAGMILive::buildSchedGraph(AAResults *AA) {
  // 遍历基本块中的调度单元
  for (SUnit &SU : SUnits) {
    MachineInstr &MI = *SU.getInstr();
    
    // 若当前指令是等待异步拷贝组完成的 Barrier 指令 (如 wait_group 0)
    if (TII->isAsyncWaitOp(MI)) {
      // 1. 将该节点与之前所有的异步加载指令建立 Token 依赖边
      for (SUnit *AsyncLoadSU : PendingAsyncLoads) {
        SU.addPred(SDep(AsyncLoadSU, SDep::Order));
      }
      // 2. 为后续所有读取 Shared Memory 的计算指令建立前驱依赖
      PendingAsyncLoads.clear();
      LastWaitSU = &SU;
    } else if (MI.mayLoad() && isSharedMemoryAccess(MI)) {
      if (LastWaitSU)
        SU.addPred(SDep(LastWaitSU, SDep::Order)); // 严禁跨越 Wait 提前读取
    }
  }
}
```

4. **逐行逻辑深度拆解**：  
   1. **隐式异步资源拓扑锁死**：将 `cp.async.commit_group` 与 `cp.async.wait_group` 抽象为 DAG 节点的 `SDep::Order` 屏障边。  
   2. **防范 RAW 静默数据损坏**：若缺少 `SDep::Order` 约束，列表调度器为了填充发射空槽，极易将读取 Shared Memory 的消费指令重排到 `wait_group` 之前发射。由于 DMA 硬件异步拷贝仍在进行，计算核心将直接从 SRAM 读取到未完成写入的历史旧数据，造成静默数据破坏。

# **5\. Post-RA 调度器与 GPU 延迟掩盖**

## **题目描述**

1. 在物理寄存器已固定的 Post-RA 指令调度阶段，调度器面临着隐藏访存/计算延迟与防止物理寄存器冲突的双重博弈。Post-RA 调度器如何实现“压力感知调度”以最大化延迟掩盖？

## **源码级深度剖析**

2. **源码文件直达**：  
   1. `llvm/lib/CodeGen/PostRASchedulerList.cpp`  
3. **核心类与核心代码块逻辑**：

在 `PostRASchedulerList.cpp` 中，物理寄存器绑定后的 Anti-dependence（WAR）边被严格建构：

```c
// llvm/lib/CodeGen/PostRASchedulerList.cpp 核心逻辑精选
void PostRASchedulerList::buildSchedGraph(AAResults *AA) {
  // 物理寄存器 Anti-dependence 追踪映射表
  std::vector<SUnit *> PhysRegDefs(TRI->getNumRegs(), nullptr);
  std::vector<SmallVector<SUnit *, 2>> PhysRegUses(TRI->getNumRegs());

  for (SUnit &SU : SUnits) {
    MachineInstr &MI = *SU.getInstr();
    // 1. 扫描当前指令读取的物理寄存器
    for (const MachineOperand &MO : MI.operands()) {
      if (MO.isReg() && MO.isUse() && MO.getReg().isPhysical()) {
        MCPhysReg Reg = MO.getReg().asMCPhysReg();
        PhysRegUses[Reg].push_back(&SU);
      }
    }
    // 2. 扫描当前指令写入的物理寄存器，建立严格的 WAR 边
    for (const MachineOperand &MO : MI.operands()) {
      if (MO.isReg() && MO.isDef() && MO.getReg().isPhysical()) {
        MCPhysReg Reg = MO.getReg().asMCPhysReg();
        for (SUnit *UseSU : PhysRegUses[Reg]) {
          if (UseSU != &SU)
            UseSU->addPred(SDep(&SU, SDep::Anti)); // 强约束：防止重写破坏尚未消费的物理寄存器
        }
        PhysRegUses[Reg].clear();
      }
    }
  }
}
```

4. **逐行逻辑深度拆解**：  
   1. **显式反依赖（Anti-dependence）防线**：在 Post-RA 阶段无法再通过虚拟寄存器重命名来解除依赖。调度器必须精准建立 `SDep::Anti` 依赖边，杜绝为了拉开指令间距以掩盖延迟而将写入指令提到读取指令之前，破坏已有物理寄存器中的有效值。  
   2. **Scoreboard 槽位延迟填充**：在不违反 WAR/WAW 约束的前提下，遍历 ReadyList，优先调度与长延迟指令不产生物理寄存器冲突的独立计算指令，把物理流水线气泡完全填满。