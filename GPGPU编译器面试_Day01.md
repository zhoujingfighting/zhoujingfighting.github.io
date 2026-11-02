# **GPGPU 编译器模拟面试与深度解析 —— Day 01**

# **1\. 寄存器压力感知调度与 Occupancy（占用率）的权衡**

## **题目描述**

在 GPGPU 架构中，SM 的寄存器文件总量固定。指令调度阶段拉大指令间距能提高 ILP，但会拉长虚拟寄存器生命周期，推高寄存器压力并损害 Occupancy。请结合 LLVM 源码分析：`MachineScheduler` 与 `RegPressureTracker` 是如何协同计算当前基本块的寄存器水位的？调度器候选比较器是如何利用 `tryPressure` 在延迟隐藏与占用率之间做权衡的？

## **源码级深度剖析**

* **源码文件直达**：  
  * `llvm/lib/CodeGen/MachineScheduler.cpp`  
  * `llvm/include/llvm/CodeGen/RegisterPressure.h`  
* **核心类与核心代码块逻辑**：

在 `llvm/lib/CodeGen/MachineScheduler.cpp` 中，`ConvergingScheduler` 在选择候选指令（`pickNodeBidirectional`）时，通过 `tryPressure` 评估候选指令对物理寄存器压力集（PSet）的影响：

```c#
// llvm/lib/CodeGen/MachineScheduler.cpp
bool ConvergingScheduler::tryPressure(const SUnit *TrySU,
                                      const SUnit *CandSU,
                                      RegPressureTracker &RPTracker,
                                      PressureChange &PChange) {
  // 1. 获取候选指令发射后导致的寄存器压力变化
  PChange = RPTracker.computePressureDiff(TrySU->getInstr());
  
  // 2. 检查是否超出目标微架构的物理寄存器上限 (Pressure Set Limit)
  unsigned CritPSet = PChange.getPSet();
  if (PChange.getUnitInc() > 0 && 
      RPTracker.getRegPressure()[CritPSet] + PChange.getUnitInc() >
      DAG->getRegPressureSetLimit(CritPSet)) {
    // 超过占用率安全阈值，直接阻断激进调度，优先选择能释放寄存器的指令
    return false;
  }
  return true;
}
```

* **逐行逻辑深度拆解**：  
  * **压力集增量推导（**`computePressureDiff`**）**：`RegPressureTracker` 检查 `TrySU` 消费的操作数（释放虚拟寄存器生命周期）与产生的定义操作数（引入新的活跃区间）。  
  * **Occupancy 红线拦截**：`DAG->getRegPressureSetLimit(CritPSet)` 返回目标 GPU（如 NVPTX/AMDGPU）给定的寄存器限额。若增量会导致单线程 VGPR 用量突破阶梯临界点，调度器立即将策略切换为减少活跃变量模式（Bottom-Up 汇聚）。

# **2\. SIMT 分支发散（Branch Divergence）与控制流结构化**

## **题目描述**

在 GPU 硬件中，发散分支依赖执行掩码（EXEC）或 SIMT 栈。请结合 LLVM 后端源码，阐述 `StructurizeCFG` Pass 是如何将不可规约流（Irreducible CFG）转换为结构化流的？Volta 引入独立线程调度（ITS）后，隐式收敛假设为何失效？

## **源码级深度剖析**

* **源码文件直达**：  
  * `llvm/lib/Transforms/Scalar/StructurizeCFG.cpp`  
* **核心类与核心代码块逻辑**：

`StructurizeCFG` 基于支配树（DominatorTree）与后支配树（PostDominatorTree）重构循环与条件分支：

```c#
// llvm/lib/Transforms/Scalar/StructurizeCFG.cpp
bool StructurizeCFG::runOnFunction(Function &F) {
  // 1. 遍历并寻找不可规约或发散的分支汇合边界
  for (BasicBlock *BB : depth_first(&F.getEntryBlock())) {
    if (order.isFlowControl(BB)) {
      // 2. 插入布尔标志 (Flow Phi) 与统一汇总块 (Flow Block)
      BasicBlock *Flow = createFlowBlock(BB);
      reorganizeControlFlow(BB, Flow);
    }
  }
  return true;
}
```

* **逐行逻辑深度拆解**：  
  * **消除多出口复杂流**：非结构化跳转被拆解为层层嵌套的单入口单出口（SESE）结构，插入统一的布尔变量记录各 Lane 的分支走向。  
  * **ITS 硬件影响**：在 Volta+ 架构上，每线程拥有独立的 PC 计数器，Warp 内 Lane 不再自然锁步执行。编译器必须显式保留汇聚标记，并插入收敛原语或屏障，防止发生数据错乱。

# **3\. Address Space（地址空间）推导与 Generic Pointer 消除**

## **题目描述**

GPU 拥有 Global、Shared/LDS、Local 等离散存储空间。Generic Pointer 需通过运行时硬件路由，性能低下。请分析 `InferAddressSpaces.cpp` 是如何通过 SSA 数据流图自动推导并替换扁平指针的？

## **源码级深度剖析**

* **源码文件直达**：  
  * `llvm/lib/Transforms/Scalar/InferAddressSpaces.cpp`  
* **核心类与核心代码块逻辑**：

`InferAddressSpacesImpl` 沿 Use-Def 链进行前向与后向搜索：

```c#
// llvm/lib/Transforms/Scalar/InferAddressSpaces.cpp
bool InferAddressSpacesImpl::run(Function &F) {
  // 1. 收集明确已知 Address Space 的根节点 (Alloca, GlobalVariable, Intrinsic)
  ValueQueue Worklist;
  for (Instruction &I : instructions(F)) {
    if (isAddressSpaceProne(&I))
      Worklist.push_back(&I);
  }

  // 2. 迭代推导 Generic Pointer 的具体存储空间
  while (!Worklist.empty()) {
    Value *V = Worklist.pop_back_val();
    unsigned InferredAS = inferAddressSpace(V);
    if (InferredAS != TargetAddrSpace::Generic) {
      rewritePointerOperands(V, InferredAS);
    }
  }
  return true;
}
```

* **逐行逻辑深度拆解**：  
  * **种子收集与追溯**：从 `alloca`（Private/Local）或架构特化全局变量（Shared/LDS）出发，沿着 `GetElementPtr` 与 `bitcast` 传播具体地址空间标记。  
  * **消除 addrspacecast**：将 Generic 指针重写为具体 `addrspace(N)` 的 `Load/Store`，消除动态硬件地址解码开销。

# **4\. 访存合并（Memory Coalescing）与向量化合流**

## **题目描述**

在 GPU 中，将标量访存打包为宽向量访存能提升总线带宽利用率。请分析 `LoadStoreVectorizer.cpp` 是如何验证访存连续性并防范别名冲突的？

## **源码级深度剖析**

* **源码文件直达**：  
  * `llvm/lib/Transforms/Vectorize/LoadStoreVectorizer.cpp`  
* **核心类与核心代码块逻辑**：

在链构建逻辑中，`LoadStoreVectorizerPass` 扫描基本块内的访存链：

```c#
// llvm/lib/Transforms/Vectorize/LoadStoreVectorizer.cpp
bool Vectorizer::isConsecutiveAccess(Value *A, Value *B) {
  const SCEV *PtrSCEVA = SE.getSCEV(A);
  const SCEV *PtrSCEVB = SE.getSCEV(B);
  // 计算两指针表达式的代数差值
  const SCEV *Diff = SE.getMinusSCEV(PtrSCEVB, PtrSCEVA);
  if (const SCEVConstant *C = dyn_cast<SCEVConstant>(Diff)) {
    // 差值必须严格等于前一访存的数据类型字节宽度
    return C->getAPInt() == ElementByteSize;
  }
  return false;
}
```

* **逐行逻辑深度拆解**：  
  * **SCEV 恒定步长分析**：利用 `ScalarEvolution` 严格判定指针偏移为常量且单调无歧义。  
  * **别名冲突阻断**：利用 `AAResults::isSafeToMove` 确保在提升或延迟合并多条指令时，不会跨越存在 `MayAlias` 的写操作，同时校验目标地址满足 16 字节硬件自然对齐。

# **5\. GPU 内存模型与 Fence 指令降级约束**

## **题目描述**

现代 GPU 支持带作用域的弱内存模型。请结合 `SelectionDAGBuilder.cpp` 剖析带有 `syncscope` 的原子操作与内存屏障是如何降级为硬件指令的？

## **源码级深度剖析**

* **源码文件直达**：  
  * `llvm/include/llvm/IR/Instructions.h`  
  * `llvm/lib/CodeGen/SelectionDAG/SelectionDAGBuilder.cpp`  
* **核心类与核心代码块逻辑**：

```c#
// llvm/lib/CodeGen/SelectionDAG/SelectionDAGBuilder.cpp
void SelectionDAGBuilder::visitFence(const FenceInst &I) {
  SyncScope::ID SSID = I.getSyncScopeID();
  AtomicOrdering Order = I.getOrdering();
  
  // 建立带 Chain 依赖的 ISD::ATOMIC_FENCE 节点，禁止内存重排
  SDValue Ops[] = { getRoot(), 
                    DAG.getTargetConstant(Order, dl, MVT::i32),
                    DAG.getTargetConstant(SSID, dl, MVT::i32) };
  DAG.setRoot(DAG.getNode(ISD::ATOMIC_FENCE, dl, MVT::Other, Ops));
}
```

* **逐行逻辑深度拆解**：  
  * **作用域与内存序映射**：`syncscope`（`workgroup`, `agent`, `system`）直接编码进 SDNode，指导后端发射带硬件作用域的指令（如 PTX 的 `fence.acq_rel.gpu`）。  
  * **Chain 边控制**：在 ScheduleDAG 中生成根依赖，强行切断屏障前后内存读写的拓扑排序可交换性。