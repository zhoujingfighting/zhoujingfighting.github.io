# **GPGPU 编译器模拟面试与深度解析 —— Day 02**

# **1\. Divergence Analysis 与 Uniformity 优化**

## **题目描述**

现代 GPGPU 引入了专门的标量/Uniform 硬件单元（如 AMD RDNA/CDNA 的 SALU/SGPR，NVIDIA Volta+ 的 Uniform Datapath）。请详细说明：LLVM 是如何结合数据流与控制流（SSA 图与 CFG）判断变量是 Uniform 还是 Divergent 的？当存在发散控制流时，为什么一个静态常量赋值或只依赖 Uniform 变量的 PHI 节点也会被标记为 Divergent？将变量识别为 Uniform 对 GPGPU 后端有哪些具体的优化收益？

## **源码级深度剖析**

* **源码文件直达**：  
  * `llvm/include/llvm/ADT/GenericUniformityImpl.h`  
  * `llvm/include/llvm/Analysis/UniformityAnalysis.h`  
* **核心类与核心代码块逻辑**：  
* 在 `llvm/include/llvm/ADT/GenericUniformityImpl.h` 中，核心类 `GenericUniformityAnalysisImpl<ContextT>` 维护发散指令集 `DivergentValues` 和发散基本块 `DivergentBlocks`。其不动点计算函数 `compute()` 核心逻辑如下：  
* 

```c
// llvm/include/llvm/ADT/GenericUniformityImpl.h
template <typename ContextT>
void GenericUniformityAnalysisImpl<ContextT>::compute() {
  // 1. 初始化自然发散源（如 threadIdx/workitem id）
  for (const auto *V : DivergentValues)
    Worklist.push_back(V);

  // 2. 基于 Worklist 推进数据依赖与控制依赖的不动点迭代
  while (!Worklist.empty()) {
    const auto *Val = Worklist.pop_back_val();

    // (A) 数据流发散传播：若输入发散，其 Users 随之发散
    for (const auto *User : Val->users()) {
      if (markDivergent(*User))
        Worklist.push_back(User);
    }

    // (B) 控制流发散传播：分支指令发散导致块变成 Divergent Block
    if (const auto *Term = dyn_cast<InstructionT>(Val)) {
      if (Term->isTerminator() && hasDivergentBranch(*Term->getParent())) {
        markBlockDivergent(*Term->getParent());
      }
    }
  }

  // 3. 同步发散（Sync Divergence）与 PHI 节点判定
  for (const auto &Block : F) {
    for (const auto &Phi : Block.phis()) {
      if (isDivergent(Phi)) continue;

      // 检查支配边界与发散前驱控制流
      bool HasDisjointPaths = false;
      for (unsigned i = 0; i < Phi.getNumIncomingValues(); ++i) {
        auto *PredBlock = Phi.getIncomingBlock(i);
        // 如果前驱块受未决发散分支控制（受发散分支的支配边界包围）
        if (isBlockDivergent(*PredBlock)) {
          HasDisjointPaths = true;
          break;
        }
      }
      if (HasDisjointPaths) {
        markDivergent(Phi); // PHI 节点因时间步调不一致产生同步发散
        Worklist.push_back(&Phi);
      }
    }
  }
}
```

* **逐行逻辑深度拆解**：  
* **1\. 自然发散源入队**：`initialize()` 匹配架构内置 Intrinsic（例如 `@llvm.amdgcn.workitem.id.x`），加入 `DivergentValues` 种子队列。  
* **2\. 数据流沿 Use-Def 不动点扩散**：只要 User 的输入操作数包含发散值，且操作本身不是收敛规约（如全 Warp Ballot），`markDivergent` 返回 true 并推入 `Worklist`。  
* **3\. Sync Divergence 机理（核心考点）**：为什么常数 PHI 也会变发散？同一 Warp 内的 32 个线程在硬件 SIMT 栈调度下分批串行进入 If-Else 分支，到达重汇聚块时，不同 Lane 在**不同物理周期**从不同前驱块流入。虽然输入值是常量（如 1 和 0），但执行生命周期的不同步导致 PHI 在汇合时对不同 Lane 赋予了不同路径的值，代码中的 `isBlockDivergent(*PredBlock)` 精确捕获了这种因控制流分歧引起的时序发散，强制将其标记为发散。  
* **4\. 后端 Codegen 收益**：证明为 Uniform 的值被指导发射为标量 ALU（`S_ADD_I32`）并分配 1 个标量物理寄存器（SGPR），节省 32 倍物理寄存器，避免触发 Occupancy 阶梯暴跌。

---

# **2\. Shared Memory (LDS) Bank Conflict 与编译期消除**

## **题目描述**

请从硬件底层与编译器优化两个维度回答：什么是 N-way Bank Conflict？硬件物理上是如何交错组织 Bank 的？在什么特殊情况下多线程访问同一 Bank 不冲突？编译器有哪些自动优化手段（如 Memory Padding、XOR Swizzling）可以消除冲突？

## **源码级深度剖析**

* **源码文件直达**：  
  * `mlir/lib/Dialect/NVGPU/Transforms/OptimizeSharedMemory.cpp`  
* **核心类与核心代码块逻辑**：  
* MLIR 在 `OptimizeSharedMemory.cpp` 中通过重写索引仿射表达式实现 XOR Swizzle：  
  * 

```c
// mlir/lib/Dialect/NVGPU/Transforms/OptimizeSharedMemory.cpp
LogicalResult SwizzleSharedMemoryPatterns::matchAndRewrite(
    memref::LoadOp loadOp, PatternRewriter &rewriter) const {
  MemRefType memrefType = loadOp.getMemRefType();
  // 仅针对 Shared Memory 空间 (MemorySpace = 3)
  if (!isSharedMemory(memrefType))
    return failure();

  Location loc = loadOp.getLoc();
  SmallVector<Value> indices = loadOp.getIndices();
  if (indices.size() < 2) return failure();

  // 获取二维访存的 Row 与 Col 索引
  Value row = indices[indices.size() - 2];
  Value col = indices[indices.size() - 1];

  // 计算 Swizzle Shift 偏移: 根据 Bank 字节数（4 字节）与向量宽度计算
  // 新列坐标: new_col = col ^ (row >> shift)
  int64_t shift = calculateSwizzleShift(memrefType);
  Value shiftedRow = rewriter.create<arith::ShRUIOp>(
      loc, row, rewriter.create<arith::ConstantIndexOp>(loc, shift));
  Value swizzledCol = rewriter.create<arith::XOrIOp>(loc, col, shiftedRow);

  // 替换列索引为 Swizzled 索引并重构 Load 指令
  indices[indices.size() - 1] = swizzledCol;
  rewriter.replaceOpWithNewOp<memref::LoadOp>(
      loadOp, loadOp.getType(), loadOp.getMemRef(), indices);
  return success();
}
```

* **逐行逻辑深度拆解**：  
* **1\. Bank 冲突物理本质**：SRAM 被划分成 32 个 Banks，4 字节交错。 **Bank\_ID \= (Byte\_Address / 4\) mod 32**。同 Warp 多线程若访问同一个 Bank 但行偏移（Word 地址）不同，Crossbar 读端口仲裁失败，触发分时串行化访存。  
* **2\. 为什么 XOR Swizzle 优于 Memory Padding**：  
  * Padding 通过将 `[32][32]` 扩展为 `[32][33]` 破坏了首地址的 16 字节（128-bit）自然对齐，使得后端无法发射 `ld.shared.v4` 等宽向量指令。  
  * XOR 洗牌利用数学性质：若列变换是双射映射（Bijection），对列地址做 `col ^ (row >> shift)` 不改变列元素的连续物理紧凑性，完美保持了 128 位对齐，且硬件 ALU 上仅增加一条单周期的 `lop3.b32` 异或指令，以近乎零的代价消除了 N-way Bank Conflict。

---

# **3\. 软件流水（Software Pipelining）与异步拷贝（Async Copy）**

## **题目描述**

在 GEMM 和卷积等高性能计算算子中，软件流水是隐藏 Global Memory 延迟的关键技术。请解释双缓冲或多级软件流水在循环编译中如何展开与重排读写依赖？传统双缓冲技术对 GPU 寄存器生命周期和 Occupancy 有什么负面影响？硬件异步拷贝特性（如 cp.async / TMA）如何改变了软件流水的编译模式？

## **源码级深度剖析**

* **源码文件直达**：  
  1. `llvm/lib/CodeGen/MachinePipeliner.cpp`  
  2. `mlir/lib/Dialect/NVGPU/Transforms/MmaToAsyncPipelining.cpp`  
* **核心类与核心代码块逻辑**：  
* 在 `llvm/lib/CodeGen/MachinePipeliner.cpp` 中，基于 Swing Modulo Scheduling (SMS) 计算发起间隔（II）并重构循环流水线阶段（Stages）：  
* 

```c
// llvm/lib/CodeGen/MachinePipeliner.cpp
bool SwingModuloScheduling::schedulePipeline(MachineLoop &L) {
  // 1. 计算资源约束下最小发起间隔 (ResMII) 与循环依赖最小间隔 (RecMII)
  computeResMII();
  computeRecMII();
  unsigned MII = std::max(ResMII, RecMII);

  // 2. 启发式搜索可行的 Initiation Interval (II)
  for (unsigned II = MII; II <= MaxII; ++II) {
    if (canSchedule(II)) {
      // 3. 构建流水展开阶段与跨迭代重排 (Prologue, Kernel, Epilogue)
      generatePipelinedLoop(II);
      return true;
    }
  }
  return false;
}

// 展开后 Steady State 核心发射循环
void SwingModuloScheduling::generatePipelinedLoop(unsigned II) {
  // Stage 0: 预取迭代 k+2 (Global Load)
  // Stage 1: 等待迭代 k+1 数据就绪 (Barrier/Wait)
  // Stage 2: 执行迭代 k 计算 (MMA / TensorCore)
}
```

* **逐行逻辑深度拆解**：  
* **1\. 传统流水线的 VGPR 暴击机制**：  
  1. 传统双缓冲数据流：`Global Memory -> VGPR -> Shared Memory`。  
  2. 当流水线深度达到 3\~4 级时，意味着有 3\~4 个 Tile 的矩阵切片数据同时处于“在途（In-flight）”状态。  
  3. 若单个 Tile 占 32 个 32-bit VGPR，多级流水会导致单线程驻留寄存器飙升至 96\~128 个。一旦跨过阶梯红线（如每个 SM 65536 个寄存器限制），SM 驻留 Warp 数直接折半，导致延迟隐藏能力全线崩溃。  
* **2\. cp.async / TMA 带来的编译重构**：  
  1. 在 MLIR `MmaToAsyncPipelining` 中，编译器将内存读写重写为专有指令 `cp.async.ca.shared.global`。  
  2. **寄存器绕行（Bypass）**：硬件 DMA 引擎直接接管数据总线将全局显存刷入片上 SRAM，完全不写回任何通用寄存器（Zero VGPR Usage for Data）。  
  3. 编译器只需要在 MIR 中维护一个极轻量的虚拟计数器（Async Token），在计算前发射 `cp.async.wait_group 0`，彻底解绑了流水线深度与通用寄存器压力的耦合。

---

# **4\. 寄存器 Bank 冲突与操作数收集器（Operand Collector）**

## **题目描述**

什么是 Register Bank Conflict？为什么现代 GPU 物理寄存器文件存在读端口限制？硬件层面的操作数收集器（Operand Collector）是如何工作的？LLVM 后端在寄存器分配和 Post-RA 指令调度阶段分别采取什么策略规避？

## **源码级深度剖析**

* **源码文件直达**：  
  * `llvm/lib/CodeGen/RegAllocGreedy.cpp`  
  * `llvm/lib/CodeGen/PostRASchedulerList.cpp`  
* **核心类与核心代码块逻辑**：  
* 在 `llvm/lib/CodeGen/RegAllocGreedy.cpp` 的 `selectOrSplit` 中，通过后端注入的提示函数规避同 Bank 着色：  
* 

```c
// llvm/lib/CodeGen/RegAllocGreedy.cpp
MCPhysReg RAGreedy::selectOrSplit(const LiveInterval &VirtReg,
                                  SmallVectorImpl<Register> &SplitVRegs) {
  // 1. 获取目标架构给出的候选物理寄存器提示集合
  SmallVector<MCPhysReg, 16> Hints;
  TRI->getRegAllocationHints(VirtReg.reg(), Order, Hints, *MF, VRM, LIS);

  // 2. 遍历候选物理寄存器，计算 Bank 冲突代价
  MCPhysReg BestPhysReg = 0;
  unsigned MinCost = ~0u;

  for (MCPhysReg PhysReg : Order) {
    // 检查物理寄存器 Bank 是否与同一指令的其他已分配操作数重叠
    unsigned ConflictCost = 0;
    for (const MachineOperand &OtherMO : getSiblingsInSameInstr(VirtReg)) {
      if (OtherMO.isReg() && OtherMO.getReg().isPhysical()) {
        // GPU 典型 4-Bank 物理分布: PhysReg % 4
        if ((PhysReg % 4) == (OtherMO.getReg().id() % 4)) {
          ConflictCost += 1000; // 施加高额 Bank 冲突惩罚权重
        }
      }
    }

    if (ConflictCost < MinCost && isAllocatable(PhysReg)) {
      MinCost = ConflictCost;
      BestPhysReg = PhysReg;
    }
  }
  return BestPhysReg;
}
```

* **逐行逻辑深度拆解**：  
* **1\. 硬件读端口争用**：SRAM 寄存器阵列按编号模 4 划分为 4 个 Bank。一条 FMA 指令需要同时读取 3 个源操作数（`src0`，`src1`，`src2`）。若它们全部位于 Bank 0，硬件单周期读端口数量不足，Operand Collector 必须分 3 个周期串行读取。  
* **2\. RegAlloc 启发式规避**：`TRI->getRegAllocationHints` 会在着色时为寄存器分配器注入拓扑约束：尽量将三操作数指令的寄存器分配在 `PhysReg % 4 != UsedBank` 的物理槽位上。  
* **3\. Post-RA 指令交换律优化**：在物理寄存器分配完毕后，若无法避免冲突，窥孔优化器扫描指令，利用代数交换律（如将 `FADD R0, R4, R1` 改写为 `FADD R0, R1, R4`），配合硬件的 Operand Reuse Cache 标志，使得复用的寄存器直接命中缓存，无需再次访问 Bank。

---

# **5\. Scoreboarding 与 Waitcnt 插入**

## **题目描述**

GPGPU 架构中对于高延迟操作（如显存加载、LDS 读写）采用异步并发执行。请对比 AMD GPU 基于显式 `s_waitcnt` 指令与 NVIDIA GPU 基于 SASS Control Code 的依赖解锁机制。编译器是如何精准插入 `s_waitcnt` 的？漏插或算错依赖会造成什么后果？

## **源码级深度剖析**

* **源码文件直达**：  
  * `llvm/lib/Target/AMDGPU/SIInsertWaitcnts.cpp`  
* **核心类与核心代码块逻辑**：  
* AMDGPU 后端在 `SIInsertWaitcnts.cpp` 中通过状态类 `WaitcntBrackets` 跟踪各类异步指令的完成计数：  
  * 

```c
// llvm/lib/Target/AMDGPU/SIInsertWaitcnts.cpp
void SIInsertWaitcnts::handleInstruction(MachineInstr &MI, 
                                         WaitcntBrackets &ScoreBrackets) {
  // 1. 检查当前指令是否读取了处于在途异步状态的寄存器 (RAW Hazard)
  AMDGPU::Waitcnt RequiredWait;
  for (const MachineOperand &MO : MI.explicit_uses()) {
    if (!MO.isReg()) continue;
    Register Reg = MO.getReg();

    // 查询该寄存器最后一次被异步指令写入时的序号 (Score)
    int RegScore = ScoreBrackets.getRegScore(Reg, VM_CNT);
    if (RegScore > ScoreBrackets.getScoreRef(VM_CNT)) {
      // 计算必须等待的安全计数阈值: 当前在途总数 - 需要完成的序号
      unsigned SafeVmcnt = ScoreBrackets.getEventCounter(VM_CNT) - RegScore;
      RequiredWait.VmCnt = std::min(RequiredWait.VmCnt, SafeVmcnt);
    }
  }

  // 2. 若必须等待，在当前消费指令正前方插入 s_waitcnt
  if (RequiredWait.hasWait()) {
    BuildMI(*MI.getParent(), MI, MI.getDebugLoc(), TII->get(AMDGPU::S_WAITCNT))
        .addImm(encodeWaitcnt(RequiredWait));
    // 推进硬件计数器至安全水位
    ScoreBrackets.applyWaitcnt(RequiredWait);
  }

  // 3. 若当前指令发起新的异步访存，记录其目标寄存器和新事件编号
  if (isAsyncLoad(MI)) {
    ScoreBrackets.setRegScore(MI.getOperand(0).getReg(), VM_CNT, 
                              ScoreBrackets.getEventCounter(VM_CNT) + 1);
  }
}
```

* **逐行逻辑深度拆解**：  
* **1\. 计数器状态追踪**：`ScoreBrackets` 独立记录三个维度的未决事件：`VM_CNT`（全局访存）、`LGKM_CNT`（LDS 访存）、`EXP_CNT`（导出指令）。当发射一条 `BUFFER_LOAD` 时，计数器累加并在寄存器位图中更新其版本号。  
* **2\. Lazy Flush（最迟等待）策略**：算法遍历基本块，遇到消费指令时才去比对 `RegScore` 与当前 `ScoreRef`。通过计算 `SafeVmcnt`，编译器只等待恰好使目标数据就绪所需的最小计数，绝不过度等待。  
* **3\. 延迟隐藏最大化**：只要消费指令未出现，中间所有无关的算术指令、地址计算均可在内存数据在途期间并发发射，使得内存加载延迟被充分吸收。  
* **4\. 硬件级后果**：GPU 硬件在执行异步指令时是纯非阻塞的。若编译器计算错误或漏插，硬件执行单元直接从尚未写入有效数据的物理寄存器中抓取陈旧值（RAW 数据竞争），引发无法在运行时被异常捕获的**静默数据损坏（Silent Data Corruption）**。