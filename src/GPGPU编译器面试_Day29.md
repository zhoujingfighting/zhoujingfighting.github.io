# GPGPU 编译器模拟面试与深度解析 —— Day 29

## 1. AMDGPUAttributor.cpp 与跨过程（IPO）硬件属性推导及递归栈防御
### 题目描述
在现代 GPGPU 编译中，函数调用（尤其是包含深度递归或跨编译单元调用的复杂 Kernel）会迫使后端分配昂贵的私有调用栈（Private Scratch Stack），并抑制寄存器配额优化。请结合 `llvm/lib/Target/AMDGPU/AMDGPUAttributor.cpp` 源码，分析 `AMDGPUAttributor` 是如何基于 LLVM `Attributor` 不动点迭代框架，在全模块调用图（Call Graph）上自底向上推导函数属性（如 `amdgpu-no-workitem-id-x`、`amdgpu-no-dispatch-ptr`、`amdgpu-no-heap-ptr`）的？当某个叶子设备函数未读取任何硬件内置 ID 时，这一属性如何向上传播至 Kernel 入口，进而使后端在 `SITargetLowering::LowerFormalArguments` 中直接省去预加载硬件系统寄存器（User SGPRs / System SGPRs）的指令与标量寄存器开销？

### 源码级深度剖析
- **源码文件直达**：
  - `llvm/lib/Target/AMDGPU/AMDGPUAttributor.cpp`
  - `llvm/lib/Target/AMDGPU/SIISelLowering.cpp`
- **核心类与核心代码块逻辑**：
  在 `AMDGPUAttributor.cpp` 中，通过继承 `AbstractAttribute` 实现跨过程硬件隐式输入依赖消除：

```cpp
// llvm/lib/Target/AMDGPU/AMDGPUAttributor.cpp 核心逻辑精选
struct AAAMDAttributesFunction : public AAAMDAttributes {
  ChangeStatus updateImpl(Attributor &A) override {
    ChangeStatus Changed = ChangeStatus::UNCHANGED;
    const Function *F = getAssociatedFunction();

    // 1. 检查当前函数体内是否直接调用了读取硬件内置寄存器的 Intrinsic
    // 例如 @llvm.amdgcn.workitem.id.x 或 @llvm.amdgcn.dispatch.ptr
    for (const Instruction &I : instructions(*F)) {
      if (const auto *II = dyn_cast<IntrinsicInst>(&I)) {
        if (isImplicitHardwareInputIntrinsic(II->getIntrinsicID())) {
          // 发现直接依赖，撤销对应的 "amdgpu-no-*" 乐观假设
          removeAssumedAttribute(II->getIntrinsicID());
        }
      }
    }

    // 2. 遍历调用图中的所有 Callee，执行格交集 (Meet Operation)
    auto CheckCallSite = [&](AbstractCallSite ACS) {
      const Function *Callee = ACS.getCalledFunction();
      if (!Callee) {
        // 存在无法解析的间接调用，保守放弃所有硬件属性裁剪
        indicatePessimisticFixpoint();
        return false;
      }
      const auto &CalleeAA = A.getAAFor<AAAMDAttributes>(*this, IRPosition::function(*Callee));
      // 将 Callee 所需的隐式硬件输入并入当前 Caller 需求集
      intersectAssumedAttributes(CalleeAA);
      return true;
    };

    A.checkForAllCallSites(CheckCallSite, *this, true);
    return Changed;
  }
};
```

- **逐行机械执行分析**：
  1. **乐观不动点迭代（Optimistic Fixpoint Iteration）**：`Attributor` 框架在初始状态假设模块内所有函数均具备 `amdgpu-no-workitem-id-y/z`、`amdgpu-no-queue-ptr` 等属性。随后沿 Call Graph 反向传播真实依赖。
  2. **System SGPR 预加载裁剪（ABI 瘦身）**：在 AMD HSA ABI 中，硬件命令处理器（CP）在启动 Wavefront 时，会根据 Kernel 元数据将 `WorkgroupID`、`WorkitemID`、`DispatchPtr` 等系统参数预先装载进物理 SGPR 槽位，并在汇编头部生成打包解包指令（如 `v_bfe_u32` 从单个寄存器中拆解 `tid.x/y/z`）。
  3. **微架构寄存器与启动延迟收益**：一旦 `AMDGPUAttributor` 证明某 Kernel 及其全套调用子树均未使用 `tid.z` 或 `dispatch_ptr`，后端代码生成器直接跳过这些寄存器分配与解包指令，不仅节省了宝贵的标量 SGPR 数量，还缩短了 Wave 启动时的硬件上下文初始化时钟周期。

---

## 2. LiveDebugValues / VarLocBasedImpl.cpp 与 MachineScheduler 中的栈槽着色（StackSlotColoring）
### 题目描述
当 GPGPU 内核寄存器溢出（Spill）不可避免时，编译器会将超出物理上限的虚拟寄存器分配到线程私有的栈槽（FrameIndex / Scratch Memory）中。若基本块内存在数十个不同的溢出变量，且它们各自占用独立的栈槽，会导致每个线程的 Private Scratch 内存足迹暴增，进而触发硬件限制的 Wave 数量下调。请结合 `llvm/lib/CodeGen/StackSlotColoring.cpp` 源码，深度剖析 `StackSlotColoring` 是如何利用 `LiveStacks` 分析各个栈槽的生命周期干涉图（Interference Graph），并通过图着色算法将生命周期不相交的多个溢出栈槽合并复用为同一物理栈偏移的？

### 源码级深度剖析
- **源码文件直达**：
  - `llvm/lib/CodeGen/StackSlotColoring.cpp`
  - `llvm/include/llvm/CodeGen/LiveStacks.h`
- **核心类与核心代码块逻辑**：
  在 `StackSlotColoring.cpp` 中，分配器按栈槽字节尺寸降序排列并执行贪心着色合并：

```cpp
// llvm/lib/CodeGen/StackSlotColoring.cpp 核心逻辑精选
bool StackSlotColoring::ColorSlots(MachineFunction &MF) {
  // 1. 收集所有非固定栈槽 (Spill Slots) 的活跃区间 (LiveInterval)
  SmallVector<int, 16> SlotList;
  for (int FI = 0, e = MFI->getObjectIndexEnd(); FI != e; ++FI) {
    if (MFI->isSpillSlotObjectIndex(FI) && !MFI->isDeadObjectIndex(FI))
      SlotList.push_back(FI);
  }

  // 2. 按栈槽字节大小降序排序，优先为大尺寸向量溢出槽 (如 16 字节 v4f32) 分配颜色
  llvm::sort(SlotList, [&](int LHS, int RHS) {
    return MFI->getObjectSize(LHS) > MFI->getObjectSize(RHS);
  });

  // 3. 遍历每个栈槽，在已分配的物理栈槽颜色池中寻找无生命周期干涉的槽位
  for (int FI : SlotList) {
    LiveInterval &LI = LS->getInterval(FI);
    int AssignedColor = -1;
    for (unsigned Color = 0; Color < ColorIntervals.size(); ++Color) {
      // 检查目标物理栈槽的对齐度与尺寸是否能容纳当前栈槽，且活跃区间无重叠
      if (canMergeSlots(Color, FI) && !LI.overlaps(ColorIntervals[Color])) {
        AssignedColor = Color;
        ColorIntervals[Color].MergeSegmentsInAsValue(LI, ...);
        break;
      }
    }
    // 4. 将虚拟 FrameIndex 映射重命名为目标物理栈槽颜色
    SlotMapping[FI] = (AssignedColor != -1) ? AssignedColor : allocateNewColor(FI, LI);
  }
  return true;
}
```

- **逐行机械执行分析**：
  1. **构建栈槽活跃区间（`LiveStacks`）**：在寄存器分配器插入 `Spill` 存储与 `Reload` 加载指令后，`LiveStacks` Pass 遍历机器指令，精确计算每个 `FrameIndex` 从首次写入到最后一次读取的指令槽位区间（`SlotIndex`）。
  2. **大尺寸优先对齐着色**：`llvm::sort` 确保 16 字节对齐的向量溢出槽优先占据栈底基准位置，随后 4 字节标量溢出槽可以直接嵌入前者生命周期结束后的空闲时间窗口中（Sub-slot Sharing）。
  3. **降低 Scratch Footprint 与提高 L1 Cache 命中率**：在 GPU 微架构中，每个线程的 Scratch 栈空间在物理显存中按 Wave 交错排布（Interleaved）。通过栈槽着色将单线程栈帧从 256 字节压缩至 64 字节，不仅将全局显存的 Scratch 分配配额削减了 75%，更使得所有溢出读写集中落在同一条 128 字节 L1 数据缓存行内，大幅缓解了溢出导致的带宽抖动。

---

## 3. NVPTXISelDAGToDAG.cpp 中的纹理/表面指令匹配与非缓存只读加载（LDG.E.NC）推导
### 题目描述
在 NVIDIA GPU 架构（Kepler 及之后）中，全局内存加载分为走标准 L1/L2 数据缓存的常规加载（`ld.global`）与走只读纹理缓存/统一缓存路径的非一致性加载（`ld.global.nc`，对应 C/C++ 中的 `__ldg()` 内建函数）。请结合 `llvm/lib/Target/NVPTX/NVPTXISelDAGToDAG.cpp` 与 `llvm/lib/Analysis/MemoryDependenceAnalysis.cpp` 源码，分析编译器是如何结合 LLVM IR 的 `invariant.load` 元数据与 `noalias` + `readonly` 函数参数属性，在指令选择阶段自动将普通 `LoadSDNode` 降级为 `NVPTX::LDG_G_f32`（即 `ld.global.nc`）的？为什么在存在潜在写别名（MayAlias）的指针上错误生成 `ld.global.nc` 会引发严重的数据一致性灾难？

### 源码级深度剖析
- **源码文件直达**：
  - `llvm/lib/Target/NVPTX/NVPTXISelDAGToDAG.cpp`
  - `llvm/lib/Target/NVPTX/NVPTXISelLowering.cpp`
- **核心类与核心代码块逻辑**：
  在 `NVPTXISelDAGToDAG.cpp` 中，指令选择器在匹配全局内存读取节点时校验只读不变性标志：

```cpp
// llvm/lib/Target/NVPTX/NVPTXISelDAGToDAG.cpp 核心逻辑精选
bool NVPTXDAGToDAGISel::SelectLDG(SDNode *N) {
  auto *LD = cast<LoadSDNode>(N);
  // 1. 确认目标操作位于全局地址空间 (AddressSpace::Global = 1)
  if (LD->getAddressSpace() != NVPTX::AddressSpace::Global)
    return false;

  // 2. 校验是否具备不可变性标记 (!invariant.load) 或只读无别名证明 (isInvariant())
  if (!LD->isInvariant()) {
    return false; // 存在写入修改可能，必须走强一致性标准 L1/L2 路径
  }

  // 3. 校验地址对齐度与向量宽度，发射带 .nc (Non-Coherent) 修饰符的硬件加载指令
  SDLoc DL(N);
  unsigned Opcode = getLDGOpcode(LD->getMemoryVT(), LD->getAlign());
  SDValue Ops[] = { LD->getBasePtr(), LD->getChain() };
  ReplaceNode(N, CurDAG->getMachineNode(Opcode, DL, LD->getVTList(), Ops));
  return true;
}
```

- **逐行机械执行分析**：
  1. **前端/中端不变性证明链**：当开发者在 Kernel 参数上同时标注 `const T* __restrict__ ptr` 时，Clang 前端为该指针生成 `readonly` 与 `noalias` 属性；中端 `AliasAnalysis` 确认整个 Kernel 生命周期内无任何 Store 节点与该基地址存在别名重叠，从而在 `LoadInst` 上附加 `!invariant.load` 元数据（即 `LD->isInvariant() == true`）。
  2. **非一致性缓存微架构物理隔离**：早期 NVIDIA 架构中，纹理缓存（Texture Cache / Read-Only Data Cache）与常规 L1 数据缓存相互独立，且**纹理缓存不监听（No Snooping）来自 SM 写入单元的缓存失效（Invalidate）信号**。即使在现代合并 L1 架构中，`.nc` 修饰符仍指示内存控制器采用非一致性缓存行锁定策略，并允许非对齐/二维空间局部性预取优化。
  3. **数据撕裂风险（Stale Cache Read）**：若指针未满足全局只读条件（例如同一 Kernel 内部先向 `ptr[idx]` 写入新值，随后通过另一别名指针读取），而编译器错误发射了 `ld.global.nc`，读请求将直接命中硬件非一致性缓存中的陈旧脏数据，绕过 L2 一致性仲裁点，导致不可预测的数值错误。

---

## 4. MLIR `AffineVectorize.cpp` 中的超向量化（Super-Vectorization）与多维循环投影映射
### 题目描述
传统的向量化（如 LLVM LoopVectorizer）通常仅针对最内层一维循环展开。然而 GPGPU 架构（如 Tensor Core 或宽向量加载指令）天然适合多维数据块映射。请结合 `mlir/lib/Dialect/Affine/Transforms/SuperVectorize.cpp` 源码，分析 MLIR 的超向量化 Pass 是如何通过 `VectorizationStrategy` 将嵌套的 `affine.for` 循环直接映射为二维甚至三维的硬件虚拟向量（如 `vector<4x8xf32>`）的？在处理带有非单位步长（Non-unit Stride）或广播维度的仿射访存时，`vectorizeAffineLoad` 如何构造 `AffineMap` 置换矩阵（Permutation Map）以对齐内存布局？

### 源码级深度剖析
- **源码文件直达**：
  - `mlir/lib/Dialect/Affine/Transforms/SuperVectorize.cpp`
  - `mlir/include/mlir/Dialect/Affine/Utils.h`
- **核心类与核心代码块逻辑**：
  在 `SuperVectorize.cpp` 中，转换器根据选定的向量化维度列表重写访存索引映射：

```cpp
// mlir/lib/Dialect/Affine/Transforms/SuperVectorize.cpp 核心逻辑精选
Operation *vectorizeAffineLoad(AffineLoadOp loadOp,
                               VectorizationState &state) {
  OpBuilder b(loadOp);
  Location loc = loadOp.getLoc();
  MemRefType memRefType = loadOp.getMemRefType();

  // 1. 构建从循环迭代空间到高维向量维度的投影置换映射 (Permutation Map)
  // 例如: 嵌套循环 (%i, %j) 访问 A[%i, %j] -> 映射为 vector<4x8xf32>
  // 若访问 A[%i, 0] (沿 j 轴广播) -> 置换映射为 (d0, d1) -> (d0, 0)
  AffineMap permutationMap = makePermutationMap(
      b.getInsertionBlock(), loadOp.getIndices(), state.strategy->loopToVectorDim);

  // 2. 构造多维向量类型并生成结构化读取算子 vector.transfer_read
  auto vecType = VectorType::get(state.strategy->vectorSizes, memRefType.getElementType());
  Value padValue = b.create<arith::ConstantOp>(loc, b.getZeroAttr(memRefType.getElementType()));

  return b.create<vector::TransferReadOp>(
      loc, vecType, loadOp.getMemRef(), loadOp.getIndices(),
      AffineMapAttr::get(permutationMap), padValue, /*inBounds=*/ArrayAttr());
}
```

- **逐行机械执行分析**：
  1. **多维循环折叠（Multi-Loop Collapse into Super-Vector）**：`SuperVectorize` 不局限于最内层循环，而是同时选取外层循环 `i`（向量宽度 4）和内层循环 `j`（向量宽度 8），直接将两层标量循环坍缩为单条操作 `vector<4x8xf32>` 的张量指令，极大保留了二维空间局部性。
  2. **置换映射（Permutation Map）与广播建模**：当内存访问索引不包含某个向量化维度（如 `A[%i, 0]` 忽略了 `%j`），`makePermutationMap` 自动在对应位置填入常量 `0` 维度（Broadcast Dimension）。下游 `VectorToGPU` 降级 Pass 识别到该零维投影后，不会生成跨步内存加载，而是发射单次标量/一维读取后在通用寄存器内部执行免费广播（Register Broadcast）。
  3. **与 GPU 线程拓扑的自然对接**：生成的 `vector<4x8xf32>` 携带完整的高维几何信息，在后续降级至 NVVM/ROCDL 时，可直接按行拆分为 4 条 128-bit 宽指令（`ld.global.v4.b64`）或直接喂给 `vector.contract` 匹配 Tensor Core 矩阵切片。

---

## 5. Triton / AI 编译器中的 Split-K 归约分解与跨 Thread Block 原子/二阶段同步机制
### 题目描述
在大语言模型推理的 Decode 阶段（Batch Size 极小，如 $M=1, N=4096, K=16384$），传统的二维空间分块（仅沿 $M$ 和 $N$ 维度切分 Grid）只能产生极少量的 Thread Block（例如仅 $4096 / 128 = 32$ 个 Block），导致拥有上百个 SM 的现代 GPU 出现严重的 **SM 算力饥饿（Low SM Utilization）**。请结合 Triton / Cutlass 编译器架构分析：编译器是如何通过 **Split-K（沿归约轴 $K$ 切分并行度）** 变换将单个输出 Tile 拆分为多个子 Block 并发计算的？在完成局部累加后，编译器在图级降级时如何权衡 **In-Place 全局原子累加（Atomic Reduction）** 与 **二阶段确定性归约内核（Two-Stage Deterministic Reduction）** 的底层代价与数值确定性（Determinism）？

### 源码级深度剖析
- **核心架构直达**：
  - `triton/python/triton/ops/matmul.py` & `triton/lib/Dialect/TritonGPU/`
  - Cutlass `GemmSplitKParallel` / `StreamK` 调度器架构模型
- **Split-K 网格映射与归约路径生成伪代码**：

```cpp
// 典型的 Split-K 算子图级变换与归约分流伪代码
void emitSplitKMatmul(OpBuilder &b, Location loc, int M, int N, int K,
                      int splitKFactor, bool requireDeterminism) {
  // 1. 将 Grid 扩展为三维拓扑: (Grid_M, Grid_N, splitKFactor)
  Value splitIdx = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::z);
  Value kChunkSize = b.create<arith::ConstantIntOp>(loc, K / splitKFactor, 32);
  Value kStart = b.create<arith::MulIOp>(loc, splitIdx, kChunkSize);
  Value kEnd = b.create<arith::AddIOp>(loc, kStart, kChunkSize);

  // 2. 每个子 Block 仅在局部区间 [kStart, kEnd) 内执行 Tensor Core 矩阵乘累加
  Value partialAcc = emitSubKLoopMMA(b, loc, kStart, kEnd);

  // 3. 归约写回路径选择 (Atomic Reduction vs Deterministic Two-Stage)
  if (!requireDeterminism) {
    // 路径 A (单内核原子归约): 直接向全局输出矩阵 C 发射硬件浮点原子加
    // 映射为 PTX: red.global.add.f32 或 atom.global.add.f32 (无需额外显存缓冲)
    emitVectorizedAtomicAdd(b, loc, globalOutPtr, partialAcc);
  } else {
    // 路径 B (二阶段确定性归约): 写入临时工作区 Workspace[splitIdx, M, N]
    // 随后由编译器自动派生第二个 Reduction Kernel 按固定顺序累加，保证严格位一致
    Value workspacePtr = computeWorkspaceOffset(b, loc, splitIdx, pidM, pidN);
    emitCoalescedVectorStore(b, loc, workspacePtr, partialAcc);
  }
}
```

- **逐行机械执行分析**：
  1. **释放 $K$ 轴并行度**：通过将 `blockIdx.z` 绑定到 $K$ 维切片，原本仅启动 32 个 Block 的瘦长矩阵乘瞬间扩展为 $32 	imes 8 = 256$ 个并发 Block，一举填满整张 GPU 的所有 SM 计算核心，将硬件利用率从 20% 拉升至 90% 以上。
  2. **路径 A（全局原子归约）的硬件瓶颈**：在 Ampere/Hopper 架构上，虽然 L2 Cache 支持原生的 FP32/FP16 硬件原子加，但当多个 `splitIdx` Block 恰巧在同一时刻向相同的输出地址发射写请求时，会引发 L2 缓存行原子锁争用（Atomic Contention），且由于多线程抵达顺序随机，浮点加法结合律失效，导致每次运行结果存在微小的非确定性误差（Non-Determinism）。
  3. **路径 B（二阶段归约与 Stream-K 演进）**：当算子要求严格确定性（如科学计算或模型训练对齐）时，编译器分配高带宽临时缓冲区 `Workspace`，第一阶段所有子 Block 全速合并写入（100% Coalesced Store），第二阶段通过派生归约内核按固定顺序求和。更先进的 **Stream-K** 算法则进一步打破 $K$ 轴的刚性整除限制，按 SM 数量将总 MAC 计算量精确均分给各个 SM，在消除尾部效应的同时将原子冲突降到最低。
