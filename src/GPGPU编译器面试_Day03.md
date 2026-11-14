# **GPGPU 编译器模拟面试与深度解析 —— Day 03**

# **1\. 算子融合与 Loop Fusion 依赖合法性分析**

## **题目描述**

在深度学习框架与 AI 编译器（如 TVM, XLA, MLIR）中，算子融合（Operator Fusion）是消除 HBM 访问开销的关键技术。算子融合在底层通常体现为循环融合（Loop Fusion）。请结合 LLVM 源码阐述：编译器如何借助 `DependenceInfo` 计算距离向量（Distance Vector）和方向向量（Direction Vector）以证明循环融合合法性？

## **源码级深度剖析**

1. **源码文件直达**：  
   1. `llvm/lib/Transforms/Scalar/LoopFuse.cpp`  
   2. `llvm/include/llvm/Analysis/DependenceAnalysis.h`  
2. **核心类与核心代码块逻辑**：  
   1. 在 `llvm/lib/Transforms/Scalar/LoopFuse.cpp` 中，`LoopFuser` 类通过依赖分析引擎检验两循环合并后是否逆转了原有的数据流时序：

```c
// llvm/lib/Transforms/Scalar/LoopFuse.cpp
bool LoopFuser::isLegalFusion(const FusionCandidate &FC0,
                              const FusionCandidate &FC1) {
  // 1. 遍历两循环间的所有内存读写依赖
  for (Instruction *I0 : FC0.MemInsts) {
    for (Instruction *I1 : FC1.MemInsts) {
      if (!I0->mayReadOrWriteMemory() || !I1->mayReadOrWriteMemory())
        continue;
      
      // 2. 调用 DependenceInfo 分析依赖向量
      if (auto D = DI.depends(I0, I1, true)) {
        // 如果存在写后读(RAW)且融合会导致负向依赖向量 (Direction::Anti)
        if (D->isLoopIndependent() && !canPreserveDependency(D.get()))
          return false;
      }
    }
  }
  return true;
}
```

3. **逐行逻辑深度拆解**：  
   1. **1\. 依赖向量字典序判定**：`DI.depends` 基于 GCD 测试和 Banerjee 不等式推导距离向量 \$\\vec{d}\$。融合合法当且仅当融合后的循环依赖向量在字典序上非负，最左侧非零分量必须为正（`>`）。若变为负（`<`），意味着依赖目标在依赖源之前执行，造成数据竞争。  
   2. **2\. GPU 专属 Cost Model 惩罚**：融合虽减少中间 Tensor 显存搬运，但显著拉长了跨算子虚拟寄存器生命周期。若估算出的 Register 需求突破 Occupancy 阶梯阈值，Pass 强制回退并选择 Tile-level Shared Memory Fusion。

# **2\. MLIR Dialect 体系与 One-Shot Bufferization 机制**

## **题目描述**

MLIR 的函数式 `tensor` 必须通过 Bufferization 转换为命令式的 `memref`。请结合 MLIR 源码，剖析 `OneShotAnalysis.cpp` 是如何通过 Alias Analysis 与 RaW 冲突检测决定内存 In-place 复用的？

## **源码级深度剖析**

4. **源码文件直达**：  
   1. `mlir/lib/Dialect/Bufferization/Transforms/OneShotAnalysis.cpp`  
   2. `mlir/include/mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h`  
5. **核心类与核心代码块逻辑**：  
   1. 在 `OneShotAnalysis.cpp` 中，`OneShotAnalysisState` 驱动 RaW 冲突排查：

```c
// mlir/lib/Dialect/Bufferization/Transforms/OneShotAnalysis.cpp
bool OneShotAnalysisState::isInPlaceMemoryWrite(OpOperand &opOperand,
                                                const AnalysisState &state) const {
  // 1. 获取该操作数写入对哪些上游 Value 产生别名影响
  AliasingValueList aliases = state.getAliasingValues(opOperand);
  
  // 2. 检查是否存在读后写冲突 (Read-After-Write Conflict)
  for (const AliasingValue &alias : aliases) {
    for (OpOperand *use : state.findConflictingReads(alias.value, opOperand)) {
      if (state.hasReadBeforeWriteConflict(use, opOperand))
        return false; // 存在冲突，强制 Out-of-place，分配新内存
    }
  }
  return true; // 无冲突，安全执行 In-place 复用
}
```

6. **逐行逻辑深度拆解**：  
   1. **1\. AliasSet 追踪**：`findConflictingReads` 沿着逆向 SSA Use-Def 链追溯 Tensor 根源。若 Tensor \$T\$ 准备在 step \$K\$ 原地覆盖，但在此之后仍有未决读取依赖旧版数据，则判定冲突成立。  
   2. **2\. In-place 复用与 GPU 降级**：判定成功后，直接重用输入端 `memref` 指针；判定失败时生成显式 `memref.alloc()` 并插入拷贝指令。

# **3\. Register Pressure 驱动的 Spill 与 Rematerialization**

## **题目描述**

在 GPU 上发生高寄存器压力时，Spill 到 Local Memory 会带来严重的内存延迟。请结合 `InlineSpiller.cpp` 剖析：分配器是如何利用 `reMaterializeFor` 评估重计算收益并消除显存读写的？

## **源码级深度剖析**

7. **源码文件直达**：  
   1. `llvm/lib/CodeGen/InlineSpiller.cpp`  
   2. `llvm/lib/CodeGen/CalcSpillWeights.cpp`  
8. **核心类与核心代码块逻辑**：

```c
// llvm/lib/CodeGen/InlineSpiller.cpp
bool InlineSpiller::reMaterializeFor(LiveInterval &VirtReg,
                                     MachineInstr &MI) {
  // 1. 查询目标指令是否满足重计算条件 (如常量加载、简单的基地址加法)
  MachineInstr *OrigDef = VRM.getOriginalDef(VirtReg.reg());
  if (!OrigDef || !TII.isRematerializable(*OrigDef))
    return false;

  // 2. 在消费指令前就地克隆发射新定义指令
  MachineBasicBlock &MBB = *MI.getParent();
  TII.reMaterialize(MBB, MI, VirtReg.reg(), OrigDef->getDebugLoc(), *OrigDef, *TRI);

  // 3. 擦除该点对待溢出全局寄存器的消费依赖
  LIS.RemoveMachineInstrFromMaps(MI);
  return true;
}
```

9. **逐行逻辑深度拆解**：  
   1. **1\. `isRematerializable` 判定**：检查定义指令是否为无侧效应、无复杂内存读取的纯算术节点（如 `MOV32ri`、标量偏移加法）。  
   2. **2\. 消解全局活跃区间**：直接在每个使用点重新发射指令，彻底斩断虚拟寄存器跨越整个基本块或循环体的活跃区间，完全避免写往 Local Memory DRAM。

# **4\. Cooperative Groups 与跨块同步（Cluster Barrier）**

## **题目描述**

现代 GPU 支持跨 Block 集群同步（Cluster-level Synchronization）。请结合 LLVM 后端剖析：带有 `syncscope("cluster")` 的屏障如何映射并在 `ScheduleDAG` 中建立依赖？

## **源码级深度剖析**

10. **源码文件直达**：  
    1. `llvm/include/llvm/IR/Instructions.h`  
    2. `llvm/lib/CodeGen/SelectionDAG/ScheduleDAGSDNodes.cpp`  
11. **核心类与核心代码块逻辑**：

```c
// llvm/lib/CodeGen/SelectionDAG/ScheduleDAGSDNodes.cpp
void ScheduleDAGSDNodes::BuildSchedUnits() {
  // 遍历所有 SDNode 节点
  for (SDNode &N : DAG->allnodes()) {
    if (N.isMachineOpcode() && TII->isBarrier(N.getMachineOpcode())) {
      SUnit *BarrierSU = getSUnit(&N);
      // 为屏障节点与前序所有带有内存侧效应的指令建立 Chain / Glue 边
      for (SUnit &PredSU : SUnits) {
        if (PredSU.hasPhysRegDefs || PredSU.mayLoad || PredSU.mayStore)
          AddPred(BarrierSU, SDep(&PredSU, SDep::Order));
      }
    }
  }
}
```

12. **逐行逻辑深度拆解**：  
    1. **1\. 拓扑序强锁死**：屏障节点打上 `SDep::Order`（Chain 依赖），强制所有前驱读写在拓扑排序中先于屏障发射，后续访存必须依赖屏障输出。  
    2. **2\. ISA 映射**：映射至 PTX 的 `barrier.cluster.wait` 或 AMD 的特定簇同步原语。

# **5\. GPU 弱内存模型与 Release Consistency 编译**

## **题目描述**

GPU 采用分级弱内存模型。编译器在降级带有 `memory_order_release` 与 `acquire` 语义的操作时，底层是如何插入缓存失效与写回指令的？

## **源码级深度剖析**

13. **源码文件直达**：  
    1. `llvm/lib/Target/NVPTX/NVPTXISelLowering.cpp`  
    2. `llvm/lib/Target/AMDGPU/SIISelLowering.cpp`  
14. **核心类与核心代码块逻辑**：

```c
// llvm/lib/Target/AMDGPU/SIISelLowering.cpp
SDValue SITargetLowering::lowerATOMIC_STORE(SDValue Op, SelectionDAG &DAG) const {
  auto *Store = cast<AtomicSDNode>(Op);
  // 若包含 Release 语义，在写入前插入 L1 缓存写回 (Writeback)
  if (Store->isRelease()) {
    SDValue FlushL1 = DAG.getNode(AMDGPUISD::BUFFER_WBINVL1, dl, MVT::Other, Store->getChain());
    // 重建 Chain 确保写回在 Store 之前完成
    return DAG.getStore(FlushL1, dl, Store->getValue(), Store->getBasePtr(), Store->getMemOperand());
  }
  return Op;
}
```

15. **逐行逻辑深度拆解**：  
    1. **1\. Release 端**：插入 `BUFFER_WBINVL1` 或 PTX `.release` 修饰符，强行将本地 SM 的 L1 脏缓存刷入跨 SM 可见的全局 L2/HBM。  
    2. **2\. Acquire 端**：配合原子读取生成 Invalidate 指令，清空本地陈旧缓存行，强制从 L2 重载最新数据。