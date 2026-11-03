# GPGPU 编译器模拟面试与深度解析 —— Day 17

## 1. RegisterCoalescer.cpp 中的子寄存器消除与跨 Lane 虚拟寄存器消除
### 题目描述
在 GPU 代码生成中，宽向量解构（Unmerge）或拼接（Merge）通常会产生大量伪指令（如 TargetOpcode::REG_SEQUENCE 和 EXTRACT_SUBREG）以及两两互拷的复制指令（COPY）。请结合 llvm/lib/CodeGen/RegisterCoalescer.cpp 源码，分析 RegisterCoalescer::joinCopy 的执行逻辑：分配器是如何利用 LaneBitmask 检验子寄存器复制能否安全合并的？为什么在 GPU 架构中盲目凝聚一个带有交叉重写（Cross-overlap）的超寄存器可能导致死锁图或引入更严重的物理寄存器碎片？

### 源码级深度剖析
- **源码文件直达**：
  - `llvm/lib/CodeGen/RegisterCoalescer.cpp`
  - `llvm/include/llvm/CodeGen/TargetRegisterInfo.h`
- **核心类与核心代码块逻辑**：
```cpp
// llvm/lib/CodeGen/RegisterCoalescer.cpp 核心逻辑精选
bool RegisterCoalescer::joinCopy(MachineInstr *CopyMI, bool &Again) {
  Register SrcReg = CopyMI->getOperand(1).getReg();
  Register DstReg = CopyMI->getOperand(0).getReg();
  unsigned SrcSubIdx = CopyMI->getOperand(1).getSubReg();
  unsigned DstSubIdx = CopyMI->getOperand(0).getSubReg();

  // 1. 获取两端虚拟寄存器所绑定的子寄存器掩码
  LaneBitmask SrcMask = TRI->getSubRegIndexLaneMask(SrcSubIdx);
  LaneBitmask DstMask = TRI->getSubRegIndexLaneMask(DstSubIdx);

  // 2. 检查两虚拟寄存器在此子通道上是否存在活跃区间干涉
  LiveInterval &SrcLI = LIS->getInterval(SrcReg);
  LiveInterval &DstLI = LIS->getInterval(DstReg);
  if (DstLI.hasSubRanges() || SrcLI.hasSubRanges()) {
    if (!checkSubRangeInterference(SrcLI, DstLI, SrcMask, DstMask)) {
      // 3. 安全合并：将源寄存器的定义直接重定向至目标子寄存器
      eliminateUndefCopy(*CopyMI);
      return true;
    }
  }
  return false;
}
```
- **逐行逻辑深度拆解**：
  1. **掩码投影与提取**：当遇到复制指令 `COPY VRegA:sub0, VRegB` 时，`getSubRegIndexLaneMask` 分别计算出源和目的所在的具体通道。
  2. **细粒度通道干涉检查**：`checkSubRangeInterference` 并不粗暴比对整个虚拟寄存器的全局生命周期，而是抽取两者的 `SubRange`，仅在 `SrcMask & DstMask` 存在重合的物理通道上检查区间重叠。如果源寄存器是临时的单通道解构产物，且目标通道在对应时间段内为 Undef 或无冲突，合并判定成立。
  3. **微架构与死锁风险防范**：若两个宽向量之间存在交叉子寄存器赋值（例如 `VRegA.sub0 = VRegB.sub1; VRegA.sub1 = VRegB.sub0;`），在缺乏额外物理中转的情况下盲目凝聚，会导致相互反依赖图形成环路（Cycle），破坏 DAG 拓扑排序，并在寄存器分配阶段造成不可解的着色死锁。此时 `joinCopy` 检测到循环依赖必须中止合并，保留原始复制指令。

---

## 2. InferAddressSpaces.cpp 中的数据流格模型（Lattice）与操作数克隆降级
### 题目描述
GPU 中未消除的 Flat/Generic 指针（addrspace(0)）会导致硬件在运行时执行动态地址解码。请结合 llvm/lib/Transforms/Scalar/InferAddressSpaces.cpp 源码，分析 InferAddressSpacesImpl 如何构建带有 Undef、Flat、SpecificAS 的格（Lattice）进行前向与后向不动点传播？当遇到包含多操作数的指针算术操作（如 GetElementPtrInst），若基地址属于特定空间但偏移量来自混合路径，Pass 如何通过操作数克隆（Cloning）实现安全的指令重写？

### 源码级深度剖析
- **源码文件直达**：
  - `llvm/lib/Transforms/Scalar/InferAddressSpaces.cpp`
- **核心类与核心代码块逻辑**：
```cpp
// llvm/lib/Transforms/Scalar/InferAddressSpaces.cpp 核心逻辑精选
bool InferAddressSpacesImpl::run(Function &F) {
  // 1. 初始化每个 Pointer 变量在格上的状态 (Undef -> SpecificAS -> Flat)
  ValueLatticeMap LatticeMap;
  initializeLattice(F, LatticeMap);

  // 2. 基于工作列表进行前向与后向数据流不动点迭代
  while (!Worklist.empty()) {
    Value *V = Worklist.pop_back_val();
    unsigned CurrentAS = LatticeMap[V];

    for (User *U : V->users()) {
      if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
        // 若指针算术基地址推导为非扁平空间，将其传递给结果
        unsigned SuccAS = updateLattice(LatticeMap[GEP], CurrentAS);
        if (SuccAS != LatticeMap[GEP]) {
          LatticeMap[GEP] = SuccAS;
          Worklist.push_back(GEP);
        }
      }
    }
  }

  // 3. 重写阶段：将已定标的 Flat 指令替换为具体空间的专用指令
  for (auto &Entry : LatticeMap) {
    if (Entry.second != TargetAddrSpace::Generic)
      cloneAndReplaceInstruction(Entry.first, Entry.second);
  }
  return true;
}
```
- **逐行逻辑深度拆解**：
  1. **格状态跃迁法则**：指针初始均为 `Undef`。当遇到来自特定分配（如 `alloca` -> Private/Local，全局 LDS 声明 -> Shared）时，格状态下推为 `SpecificAS`；若一条分支合流路径中混入了不同的空间（例如通过 PHI 汇聚了 Global 和 Shared 指针），状态不可逆地下跌至底部 `Flat`。
  2. **GEP 与算术节点克隆（Cloning）**：如果基地址指针被成功证明为 `addrspace(1)`（Global），但其下游 GEP 的类型签名依然是通用的 `ptr`，编译器不能直接就地修改原指令的类型，否则会破坏 LLVM SSA 类型的强校验。`cloneAndReplaceInstruction` 会根据推导出的具体空间创建一个新的 `GetElementPtrInst`，使其直接返回 `ptr addrspace(1)`，并将后续的 Load/Store 操作数重新接线到新指令上。
  3. **微架构硬件收益**：消除 Generic 指针后，底层指令发射器不再生成必须经过片上 MMU 译码电路的低速扁平指令（如 PTX `ld.f32`），而是直接映射为专用硬件指令（如 `ld.global.f32` 或 `ld.shared.f32`），彻底消除了 GPU 硬件地址路由判定的流水线停顿。

---

## 3. AMDGPUISelDAGToDAG.cpp 中的硬件直接平面原子累加（Flat Global Atomics）降级与 Cache Coherence 标记
### 题目描述
在大规模并行归约或梯度累加中，现代 GPU 提供了免旁路 L1 的全局直接原子硬件通路。请结合 llvm/lib/Target/AMDGPU/AMDGPUISelDAGToDAG.cpp 源码，分析在匹配 ISD::ATOMIC_LOAD_ADD 到机器指令（如 FLAT_ATOMIC_ADD 或 GLOBAL_ATOMIC_ADD）时，编译器如何判定是否可以应用 SystemScope 或 AgentScope？指令字中的 SLC（System Level Coherent）与 GLC（Globally Level Coherent）标志位是如何在代码生成阶段被静态置位的？

### 源码级深度剖析
- **源码文件直达**：
  - `llvm/lib/Target/AMDGPU/AMDGPUISelDAGToDAG.cpp`
  - `llvm/lib/Target/AMDGPU/SIISelLowering.cpp`
- **核心类与核心代码块逻辑**：
```cpp
// llvm/lib/Target/AMDGPU/AMDGPUISelDAGToDAG.cpp 核心逻辑精选
bool AMDGPUDAGToDAGISel::SelectGlobalAtomic(SDNode *N) {
  auto *AtomicNode = cast<AtomicSDNode>(N);
  SDLoc DL(N);

  // 1. 验证目标指针必须在全局地址空间 (addrspace 1)
  if (AtomicNode->getAddressSpace() != AMDGPUAS::GLOBAL_ADDRESS)
    return false;

  // 2. 解析系统范围 (System Scope) 与缓存一致性标志
  unsigned CPol = 0;
  if (AtomicNode->getSyncScopeID() == SyncScope::System) {
    CPol |= AMDGPU::CPol::GLC; // Globally Level Coherent: 绕过或失效 L1 Cache
  }
  if (AtomicNode->getOrdering() == AtomicOrdering::SequentiallyConsistent) {
    CPol |= AMDGPU::CPol::SLC; // System Level Coherent: 旁路 L2，由内存控制器直接仲裁
  }

  // 3. 构建发射硬件指令 GLOBAL_ATOMIC_ADD_RTN 或非返回版
  SDValue Ops[] = { AtomicNode->getChain(), AtomicNode->getBasePtr(),
                    AtomicNode->getVal(), CurDAG->getTargetConstant(CPol, DL, MVT::i32) };
  ReplaceNode(N, CurDAG->getMachineNode(AMDGPU::GLOBAL_ATOMIC_ADD_F32, DL,
                                        AtomicNode->getVTList(), Ops));
  return true;
}
```
- **逐行逻辑深度拆解**：
  1. **硬件一致性标志位（GLC / SLC）注入**：
     - `GLC`（Global Level Coherent）：若设为 1，告诉硬件该原子操作具有多核/跨 SM 可见性要求，执行时必须绕过或强制失效本 SM 内部的局部 L1 向量缓存（Vector L1），直接将原子请求投递至跨 SM 共享的 L2 缓存内部的 ALU。
     - `SLC`（System Level Coherent）：若设为 1，表示该原子操作涉及跨设备或与 Host CPU 的系统级内存一致性，指示 L2 Cache 作为非分配（No-Allocate）旁路流，直接推向外部显存控制器的硬件仲裁队列。
  2. **免返回寄存器流水线优化**：若 `AtomicNode->use_empty()` 成立，选择器直接匹配免返回版本 `GLOBAL_ATOMIC_ADD_F32`（非 `_RTN` 版本）。硬件在此模式下免除将旧值写回 VGPR 的总线通路，消除了一次物理寄存器写入开销和后序流水线停顿。

---

## 4. ConvertVectorToGPU.cpp 中的 Warp Shuffle 归约与二维子通道映射
### 题目描述
在 MLIR 的 vector 到 gpu 降级路径中，高维向量归约（如 vector.multi_reduction）需要映射到硬件 Warp 级 Shuffle 指令。请结合 mlir/lib/Conversion/VectorToGPU/VectorToGPU.cpp 源码，解释 WarpReductionRewriter 如何计算每个线程在多维归约轴上的展开步长（Stride）？在展开为 gpu.shuffle xor 序列时，Pass 如何保证多维累加不会跨越线程束物理边界（Wavefront/Warp Boundary）？

### 源码级深度剖析
- **源码文件直达**：
  - `mlir/lib/Conversion/VectorToGPU/VectorToGPU.cpp`
- **核心类与核心代码块逻辑**：
```cpp
// mlir/lib/Conversion/VectorToGPU/VectorToGPU.cpp 核心逻辑精选
LogicalResult WarpReductionRewriter::matchAndRewrite(
    vector::ReductionOp reductionOp, PatternRewriter &rewriter) const {
  Location loc = reductionOp.getLoc();
  Value val = reductionOp.getVector();
  unsigned warpSize = targetSubgroupSize; // 例如 32 (NVIDIA) 或 64 (AMD)

  // 针对 Warp 宽度进行对分步长迭代 (16, 8, 4, 2, 1)
  for (unsigned offset = warpSize / 2; offset > 0; offset /= 2) {
    Value offsetVal = rewriter.create<arith::ConstantIntOp>(loc, offset, 32);
    Value widthVal = rewriter.create<arith::ConstantIntOp>(loc, warpSize, 32);

    // 1. 发射硬件层面的跨 Lane 异或洗牌原语 (gpu.shuffle xor)
    auto shuffleOp = rewriter.create<gpu::ShuffleOp>(
        loc, val, offsetVal, widthVal, gpu::ShuffleMode::XOR);

    // 2. 发射对应类型的局部累加指令 (如 arith.addf)
    val = rewriter.create<arith::AddFOp>(loc, val, shuffleOp.getShuffleResult());
  }

  rewriter.replaceOp(reductionOp, val);
  return success();
}
```
- **逐行逻辑深度拆解**：
  1. **对分折半网络构建**：若目标 Wave/Warp 规格为 32，外部循环从 `offset = 16` 递减至 `1`，生成固定深度为 5 的严格收敛拓扑。
  2. **跨束隔离与 widthVal 作用**：`gpu::ShuffleOp` 显式接受一个 `widthVal` 参数。当一个 Thread Block 包含多个 Warp（如 128 个线程包含 4 个 Warp）时，`widthVal = 32` 保证底层生成的汇编指令（如 PTX `shfl.sync.bfly` 的 clamp 字段）将通信范围严格锁死在当前 Warp 内，禁止任何 Lane 与其他物理 Warp 交叉通信。
  3. **数据广播与最终产物**：经过 5 轮 Shuffle-Add，全 Warp 内的所有 32 个线程的寄存器中都持有最终相同的归约累加标量结果（All-Reduce），彻底规避了向片上 Shared Memory 进行中转写回再读取的数条指令延迟。

---

## 5. Triton / IREE 中的 Asynchronous Prefetch Pipeline 与共享内存三级缓冲（Triple Buffering）展开
### 题目描述
在处理极高延迟的显存加载时，现代算子编译器通常构建三级甚至四级软件流水（Multi-stage Prefetching）。请结合 Triton 编译器的流水线 Pass 分析：编译器如何将循环归纳变量拆分为 Prologue、Kernel 与 Epilogue？在构建稳态（Steady-State）循环体时，编译器如何追踪三个不同阶段（Stage 0: 正在异步 DMA 载入, Stage 1: 正在等待屏障, Stage 2: 正在送入 Tensor Core 计算）的缓冲切片地址，并静态生成无分支的循环展开？

### 源码级深度剖析
- **核心源码直达**：
  - `triton/lib/Dialect/TritonGPU/Transforms/Pipeline.cpp`
- **核心代码块逻辑**：
```cpp
// 典型的三级缓冲软件流水展开核心逻辑伪代码
void emitTripleBufferedLoop(OpBuilder &b, ForOp oldLoop, Value smemBuffers[3]) {
  // === 1. Prologue 预取阶段 ===
  // 提前向 DMA 发起第 0 和第 1 迭代的数据加载 (非阻塞)
  asyncLoadGlobalToShared(b, smemBuffers[0], loopIter = 0); // Stage 0 填充
  asyncLoadGlobalToShared(b, smemBuffers[1], loopIter = 1); // Stage 1 填充

  // === 2. Steady State 稳态循环体 ===
  b.create<scf::ForOp>(loc, start = 0, end = numIters - 2, step = 1,
                       [&](OpBuilder &b, Value iv) {
    // 模 3 环形缓冲索引指针
    Value loadBufIdx = (iv + 2) % 3; // 正在预取后续数据
    Value waitBufIdx = (iv + 1) % 3; // 正在等待传输就绪
    Value compBufIdx = (iv + 0) % 3; // 正在供 Tensor Core 计算

    // 并发动作 A: 发起下下个迭代的预取 (Stage 0)
    asyncLoadGlobalToShared(b, smemBuffers[loadBufIdx], iv + 2);

    // 并发动作 B: 等待上一次预取就绪 (Stage 1 -> Stage 2 屏障)
    awaitAsyncCopy(b, smemBuffers[waitBufIdx]);

    // 并发动作 C: 计算当前阶段数据 (Stage 2)
    mmaComputeTile(b, smemBuffers[compBufIdx]);
  });

  // === 3. Epilogue 收尾阶段 ===
  // 消费最后两轮剩余在片上缓冲中的计算数据
}
```
- **逐行逻辑深度拆解**：
  1. **流水线三阶段状态机物理映射**：
     - Stage 0（生产者发起）：调用硬件异步引擎（如 TMA / `cp.async`）发射总线读取请求，数据在途（In-Flight）。
     - Stage 1（传输推进）：数据正穿透 L2 写入 Shared Memory，由硬件记分牌或异步屏障（`mbarrier`）跟踪未决字节。
     - Stage 2（消费者计算）：等待屏障计数清零后，Tensor Core / VALU 启动 MMA 计算。
  2. **环形地址轮转（Index Modulo Sinking）**：由于片上 SRAM 预先分配了 3 组固定物理偏移的 Tile 空间，编译器将循环迭代变量映射为 `iv % 3`。通过在 IR 降级时完全展开为 3 步循环步进（Unroll factor = 3），可静态将 `(iv + k) % 3` 完全折叠为常量物理基地址，消除循环体内部所有的运行时模除与动态偏移计算。
  3. **延迟隐藏效能**：三级缓冲将片上内存需求提升了 3 倍，但彻底斩断了计算核心对显存延迟的暴露：只要 Stage 0 和 Stage 1 的搬运总延迟小于 Stage 2 的 Tensor Core 矩阵乘计算周期，GPU 计算单元便能持续维持在接近 100% 的峰值算力吞吐。
