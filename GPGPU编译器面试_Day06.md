# **GPGPU 编译器模拟面试与深度解析 —— Day 06**

# **1\. Machine Hazard Recognizer 的设计与 GPU 流水线冲突**

## **题目描述**

在 LLVM 后端开发中，TargetHazardRecognizer 的工作机制是什么？在 Post-RA（寄存器分配后）指令调度阶段，针对 GPU 复杂的流水线依赖（如数据 RAW 冲突、特定执行单元延迟、长周期访存等），编译器如何利用 Hazard Recognizer 准确判定 Hazard 并决定插入 NOP 或是设置依赖等待周期（如 SASS 中的 yield / depbar / waitcnt）？请结合源码详细阐述其设计架构与控制流程。

## **参考解析**

### **1\. 源码文件直达**

* `llvm/lib/Target/AMDGPU/AMDGPUHazardRecognizer.cpp`  
* `llvm/include/llvm/CodeGen/ScheduleHazardRecognizer.h`

### **2\. 核心类与核心代码块逻辑**

在 `AMDGPUHazardRecognizer.cpp` 中，识别器通过维护指令滑动窗口与硬件延迟表判定发射安全性：

```c
// llvm/lib/Target/AMDGPU/AMDGPUHazardRecognizer.cpp
ScheduleHazardRecognizer::HazardType
AMDGPUHazardRecognizer::getHazardType(SUnit *SU, int Stalls) {
  MachineInstr *MI = SU->getInstr();

  // 1. 检查标量读写与向量指令间的寄存器冲突 (RAW / WAW)
  if (SIInstrInfo::isVALU(*MI)) {
    int WaitStates = checkVALUHazards(MI);
    if (WaitStates > Stalls)
      return Hazard_IsHazard;
  }

  // 2. 检查特定硬件单元的结构性冲突 (如 VMEM / SMEM 软子句冲突)
  if (checkClauseHazards(MI))
    return Hazard_NoInstanceAvailable;

  return ScheduleHazardRecognizer::getHazardType(SU, Stalls);
}

void AMDGPUHazardRecognizer::EmitInstruction(SUnit *SU) {
  // 更新滑动历史窗口，若为 nullptr 则代表插入空泡推进周期
  History.push_front(SU ? SU->getInstr() : nullptr);
  if (History.size() > MaxHistoryDepth)
    History.pop_back();
}
```

### **3\. 逐行逻辑深度拆解**

* **滑动窗口追踪（History Queue）**：`History` 环形队列记录最近发射的机器指令序列，窗口大小对应微架构中最长延迟周期的硬件上限。  
* **冒险检测（Hazard Arbitration）**：当调度器准备发射下一个候选指令时，`checkVALUHazards` 扫描历史队列，若前序写入与当前读取的时间跨度小于硬件临界流水级数，返回 `Hazard_IsHazard` 强制挂起该指令。  
* **空泡填补与 NOP 发射**：若当前无独立无关指令可调度，调度器调用 `EmitInstruction(nullptr)`，后置 Pass 映射发射显式硬件空操作（如 `s_nop N`），确保物理流水线读写绝对安全。

---

# **2\. Distributed Shared Memory (DSMem) 与 Cluster 级数据交换编译**

## **题目描述**

现代 GPGPU 架构（如 NVIDIA Hopper 架构的 Threadblock Cluster）引入了分布式共享内存（Distributed Shared Memory, DSMem），允许同一个 Cluster 内的不同 SM 跨块直接访问对方的 Shared Memory。请问 LLVM IR 如何表征这种跨块内存访问与地址空间映射？针对跨 SM 的异步数据传输与同步，编译器在 ScheduleDAG 中需要建立怎样的内存栅栏与数据流图依赖？

## **参考解析**

### **1\. 源码文件直达**

* `llvm/lib/Target/NVPTX/NVPTXISelLowering.cpp`  
* `llvm/lib/CodeGen/SelectionDAG/SelectionDAGBuilder.cpp`

### **2\. 核心类与核心代码块逻辑**

在指令合法化与选择阶段，跨块集群访存通过定制的地址计算与专用 Intrinsic 生成带 Cluster 作用域的硬件操作：

```c
// 典型的 Cluster 地址转换与跨块屏障生成逻辑
SDValue NVPTXTargetLowering::LowerClusterBarrier(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  // 1. 构建带 Cluster 作用域的内存屏障节点 (TokenFactor Chain)
  SDValue Chain = Op.getOperand(0);
  SDValue ClusterBarrier = DAG.getNode(
      NVPTXISD::BARRIER_CLUSTER, DL, MVT::Other, Chain);
  
  // 2. 强制锁死调度顺序，禁止后续 Load/Store 跨越屏障
  return ClusterBarrier;
}
```

### **3\. 逐行逻辑深度拆解**

* **地址空间映射**：通过扩展地址空间指针属性（如 `addrspace(6)`）或调用 `@llvm.nvvm.mapa` 硬件映射指令，将当前 Block 内的相对 LDS 偏移加上目标 `cluster_ctaid` 转化为集群物理通用地址。  
* **ScheduleDAG 内存链约束**：在调度图中为跨块读写指令和 `cluster.barrier` 建立带有 `SDep::Order` 属性的 Chain 边，阻断拓扑重排。同时扩展别名分析（Alias Analysis），将远程指针查询显式声明为 `MayAlias`，防止死代码消除（DSE）错误擦除跨块写入。

---

# **3\. 标量化（Scalarization）与向量化（Vectorization）的动态博弈**

## **题目描述**

在 GPGPU 编译优化 Pass 中，通常倾向于将多个标量访存合并为向量访存（如将 4 个 f32 标量 Load 合并为 v4f32 / LDG.E.128 指令）以提高内存吞吐。然而，编译器有时会主动将宽向量指令拆散进行“标量化（Scalarization）”。请详细分析：编译器在什么情况下必须进行标量化？这种拆分对非对齐内存访问、Bank 冲突以及访存合并分别会产生什么影响？

## **参考解析**

### **1\. 源码文件直达**

* `llvm/lib/CodeGen/GlobalISel/LegalizerHelper.cpp`  
* `llvm/lib/Transforms/Vectorize/VectorCombine.cpp`

### **2\. 核心类与核心代码块逻辑**

在 `LegalizerHelper.cpp` 中，当指针未满足对齐硬约束时，系统强制执行拆分标量化：

```c
// llvm/lib/CodeGen/GlobalISel/LegalizerHelper.cpp
LegalizerHelper::LegalizeResult
LegalizerHelper::scalarizeVectorOp(MachineInstr &MI, unsigned OpIdx) {
  Register DstReg = MI.getOperand(0).getReg();
  LLT DstTy = MRI.getType(DstReg);
  LLT EltTy = DstTy.getElementType();

  // 1. 拆散宽向量操作数为独立的标量指令流
  SmallVector<Register, 8> ScalaredRegs;
  for (unsigned i = 0; i < DstTy.getNumElements(); ++i) {
    Register NewDst = MRI.createGenericVirtualRegister(EltTy);
    MIRBuilder.buildInstr(MI.getOpcode())
        .addDef(NewDst)
        .addUse(extractElement(MI.getOperand(1).getReg(), i));
    ScalaredRegs.push_back(NewDst);
  }

  // 2. 将标量结果重构或直接供下游标量消费
  MIRBuilder.buildBuildVector(DstReg, ScalaredRegs);
  MI.eraseFromParent();
  return Legalized;
}
```

### **3\. 逐行逻辑深度拆解**

* **非对齐异常拦截**：GPU 硬件向量加载指令（如 `LDG.E.128`）严格要求 16 字节自然对齐。若地址推导证明基址可能非对齐，强制生成向量指令将引发硬件非法对齐异常，编译器必须调用 `scalarizeVectorOp` 拆解为 4 条安全的 32-bit 标量加载。  
* **寄存器重用与 Bank 优化**：宽向量（如 `v8f32`）要求连续的物理寄存器槽位，极易在 RA 阶段引发外部碎片和 Spill。标量化将单一连续长生命周期打散为多个独立的短生命周期标量，赋予分配器跨 Bank 打散物理寄存器的自由度，消除了 Operand Collector 的读端口争用。

---

# **4\. Tile Size 选择与 AI 编译器自动调优（Auto-Tuning）代价模型**

## **题目描述**

在 Triton、TVM 等现代 AI 领域专用编译器中，Tile Size（分块大小）的选择对 GEMM 等算子性能起决定性作用。请阐述自动调优框架如何在庞大的搜索空间中高效寻找最优 Tile 形状？并详细剖析：在设计静态代价模型时，必须量化评估哪些 GPGPU 硬件层的核心指标？

## **参考解析**

### **1\. 核心算法与代价模型实现**

在调优搜索器中，静态代价模型由 Occupancy 阶梯函数与访存/算术强度方程联合驱动：

```c
// 典型的 AI 编译器 Tile 代价模型数学抽象
double evaluateTileCost(int TileM, int TileN, int TileK, const GPUSpec &Spec) {
  // 1. 物理资源约束与 SM 驻留 Block 计算 (Occupancy)
  int RegsPerThread = estimateRegUsage(TileM, TileN, TileK);
  int SmemPerBlock = calculateSmemBytes(TileM, TileN, TileK);
  int ActiveBlocks = std::min({Spec.MaxBlocksPerSM,
                              Spec.MaxRegsPerSM / (RegsPerThread * 32),
                              Spec.MaxSmemPerSM / SmemPerBlock});
  if (ActiveBlocks == 0) return 1e9; // 超出物理上限，直接剪枝

  // 2. 算术强度与访存事务合并效率评估
  double FLOPs = 2.0 * TileM * TileN * TileK;
  double BytesLoaded = (TileM * TileK + TileN * TileK) * sizeof(half);
  double CoalescingEfficiency = (TileN % 32 == 0) ? 1.0 : 0.25;

  // 综合延迟评分: 越低越优
  return (BytesLoaded / (Spec.DRAMBandwidth * CoalescingEfficiency)) + 
         (FLOPs / (Spec.PeakComputeTOPS * (ActiveBlocks / (double)Spec.MaxBlocksPerSM)));
}
```

### **2\. 逐行逻辑深度拆解**

* **硬约束空间剪枝**：若单 Block 消耗的 Shared Memory 或寄存器超出 SM 物理规格，直接丢弃该候选。  
* **Occupancy 阶梯评估**：量化计算激活的 Warps 是否达到隐藏 Global Memory 访问延迟的临界饱和线。  
* **合并访存与算术强度平衡**：评估内层连续维度是否满足 128 字节合并访存；通过 Roofline 模型判断该 Tile 处于 Memory-Bound 还是 Compute-Bound，引导搜索向更高数据复用率的几何形状收敛。

---

# **5\. Warp-Level 原语与 Mask 传播优化**

## **题目描述**

在 CUDA / GPGPU 编程中，Warp-Level 原语（如 `__shfl_sync`、`__ballot_sync`）被广泛用于线程块内的高效通信。从 LLVM 中端 IR 到后端代码生成，编译器如何对代表 Active Mask 的控制流进行分析与传播优化？如何在保证 SIMT 汇聚安全的前提下，消除冗余掩码计算？

## **参考解析**

* ### **1\. 源码文件直达**

* `llvm/lib/Transforms/Scalar/DivergenceAnalysis.cpp`  
* `llvm/include/llvm/IR/IntrinsicsNVPTX.h`

### **2\. 核心类与核心代码块逻辑**

在控制流分析 Pass 中，编译器验证收敛域与支配边界，合并相邻的掩码调用：

* 

```c
// 典型的 Warp Mask 传播优化逻辑
bool optimizeWarpMask(Function &F, DominatorTree &DT) {
  for (BasicBlock &BB : F) {
    Instruction *ActiveMaskOp = nullptr;
    for (Instruction &I : BB) {
      // 匹配获取活跃掩码的 Intrinsic (如 @llvm.nvvm.activemask)
      if (isNVVMActiveMask(I)) {
        if (ActiveMaskOp && isSafeToReplace(ActiveMaskOp, &I, DT)) {
          // 在无发散分支的支配区间内，后续掩码调用直接复用首个结果
          I.replaceAllUsesWith(ActiveMaskOp);
          I.eraseFromParent();
        } else {
          ActiveMaskOp = &I;
        }
      }
    }
  }
  return true;
}
```

### **3\. 逐行逻辑深度拆解**

* **掩码冗余消除**：在基本块内部或无发散分支的单入口单出口（SESE）区域中，所有线程处于固定的活跃集合。分析器通过支配树（DominatorTree）证明控制流收敛，将重复的 `__activemask()` 调用统一替换为首次计算的 SSA 变量，消除多余硬件读取。  
* **收敛屏障防线**：若两次 Warp 原语之间跨越了发散条件分支，活跃掩码发生物理改变，`isSafeToReplace` 返回 false 强制阻止替换，防止未就绪的非活跃 Lane 参与通信导致硬件死锁。

