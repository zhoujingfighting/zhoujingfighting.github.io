# GPGPU 编译器模拟面试与深度解析 —— Day 43：`AMDGPUMemoryLegalizer` 缓存穿透、`InlineSpiller` 支配树、`NVPTXLowerUnreachable` 与 FP8 动态量化

### 1. 【AMDGPU 内存依赖图与硬件等待计数器插入】`AMDGPUMemoryLegalizer.cpp` 中的原子操作与全局/局部内存顺序模型降级（Memory Model Legalization）

* **题目描述**：在 C++11 / OpenCL / HIP 内存模型中，带有 `Acquire`、`Release` 或 `SeqCst` 内存序（Ordering）及不同同步作用域（`SyncScope::Workgroup` vs `SyncScope::Agent/Device` vs `SyncScope::System`）的读写与原子操作（`AtomicRMW` / `AtomicCmpXchg`），必须由编译器在代码生成前转换为底层硬件可识别的缓存控制位（`GLC` / `SLC` / `DLC`）与内存屏障组合。请结合 `llvm/lib/Target/AMDGPU/AMDGPUMemoryLegalizer.cpp` 源码，深度剖析 `AMDGPUMemoryLegalizer` 是如何根据目标子架构（GFX9/CDNA vs GFX10+/RDNA）的缓存一致性拓扑（Cache Coherency Model），在 `Release` 写操作前插入 `s_waitcnt` 与缓存刷新、在 `Acquire` 读操作后插入向量 L1 缓存失效指令（`BUFFER_WBINVL1_VOL`）的？为什么在 RDNA（GFX10+）引入 CU 级共享 L0/L1 缓存后，非原子普通 `Acquire` 加载也必须携带 `GLC`（Global Level Coherent）修饰符以穿透 L0 缓存？

* **源码文件直达**：

  * `llvm/lib/Target/AMDGPU/AMDGPUMemoryLegalizer.cpp`

* **核心代码块逻辑**：

```cpp
// llvm/lib/Target/AMDGPU/AMDGPUMemoryLegalizer.cpp 核心逻辑精选
bool AMDGPUMemoryLegalizer::expandLoad(const MemOpInfo &MOI,
                                       MachineBasicBlock::iterator &MI) {
  // 1. 提取内存序 (AtomicOrdering) 与同步作用域 (SyncScope::ID: Wavefront / Workgroup / Agent / System)
  AtomicOrdering Ordering = MOI.getOrdering();
  SyncScope::ID SSID = MOI.getSyncScopeID();

  // 2. 判定当前加载是否要求跨 CU / 跨设备可见性 (例如 Agent / Device / System 级 Acquire 读):
  if (isAcquireOrStronger(Ordering) && SSID != SyncScope::SingleThread) {
    // (a) 在 RDNA (GFX10+) 架构中，每个 CU 拥有私有的 L0 Vector Cache，
    // 即使在同一个 Workgroup 内部，不同 Wave 若调度在不同子阵列上也可能存在 L0 不一致，
    // 因此必须强制设置 GLC (Global Level Coherent / DLC) 缓存控制位，迫使读请求绕过 L0/L1 穿透至 L2！
    enableCacheBypassFlags(*MI, /*GLC=*/true, /*DLC=*/isSystemScope(SSID));

    // (b) 插入等待计数 s_waitcnt vmcnt(0) 确保加载数据落入寄存器后才执行后续指令，
    // 并发射 BUFFER_WBINVL1_VOL 使当前 CU 本地向量缓存中的陈旧缓存行全部失效 (Invalidate)！
    insertWaitcntAndCacheInvalidate(MI, SSID);
    return true;
  }
  return false;
}
```

* **逐行机械执行分析**：

  1. **RDNA（GFX10+）L0 缓存与 `GLC` 位硬件仲裁**：在早期 GCN/CDNA 架构中，单个 CU 内的所有 Wavefront 共享同一个 L1 Vector Cache；而 RDNA 架构为了降低功耗和延迟，引入了更小更快的 **L0 Vector Cache**（每个 WGP 内部切分），且 L0 之间不维护自动硬件一致性嗅探。当程序执行带 `Acquire` 语义的同步读取（如自旋锁状态位读取）时，若不带 `GLC` 标记，读请求会死死命中 L0 中的旧值陷入死循环。`AMDGPUMemoryLegalizer` 强制开启 `GLC` 旁路并紧随 `BUFFER_WBINVL1_VOL`，保证读取穿透到全局一致性仲裁点（L2 Cache 或系统总线 `DLC`）。

  2. **非阻塞预取与 `Release` 写前排空**：在处理 `Release` 写操作（如 `atomic_store_explicit(&flag, 1, memory_order_release)`）时，编译器在写指令**之前**插入 `s_waitcnt vmcnt(0) lgkmcnt(0)`，强制排空本线程此前所有在途的普通数据写事务，确保 Payload 数据 100% 抵达 L2 之后，标志位更新请求才被送入总线，严格守住 `Happens-Before` 因果律。

---

### 2. 【LLVM 向量寄存器合并与子通道精确活跃性】`InlineSpiller.cpp` 中的溢出代码下沉（Spill Hoisting / Sinking）与重物质化链式折叠

* **题目描述**：在贪心寄存器分配器（`RegAllocGreedy`）被迫做出溢出决策后，`InlineSpiller` 负责在 IR/MIR 中插入具体的内存存储（Spill）与加载（Reload）指令。然而，如果一个被溢出的虚拟寄存器在多个相邻分支或循环内部被频繁读取，简单的“在每次使用前都插一条 `Reload`”会导致严重的指令膨胀与带宽浪费；反之，如果把 `Store` 放在所有分支之前，又可能在从不走溢出路径的快速分支上白白执行内存写入。请结合 `llvm/lib/CodeGen/InlineSpiller.cpp` 源码，深度剖析 `InlineSpiller::hoistSpill` 与 `insertReloads` 是如何基于支配树（`MachineDominatorTree`）寻找最近公共支配块（Nearest Common Dominator）合并多次相邻 Reload、并将 Spill Store 下沉到真正需要它的冷分支路径中的？在执行重物质化链式回溯（`traceSiblingValue`）时，分配器如何穿透多层 `COPY` 与区间切分（Live Range Split）找到最原始的廉价定义指令？

* **源码文件直达**：

  * `llvm/lib/CodeGen/InlineSpiller.cpp`

* **核心代码块逻辑**：

```cpp
// llvm/lib/CodeGen/InlineSpiller.cpp 核心逻辑精选
void InlineSpiller::hoistSpill(LiveInterval &OrigLI, SmallVectorImpl<LiveInterval *> &NewIntervals) {
  // 1. 收集所有需要插入 Spill Store 的基本块位置
  // 假设一个虚拟寄存器在入口块定义，但仅在两个冷分支块 (ColdBB1, ColdBB2) 中发生局部溢出使用，
  // 而在占 95% 执行概率的主干热路径 (HotBB) 中全程留在物理寄存器中未被溢出！
  SmallPtrSet<MachineBasicBlock *, 8> SpillBlocks;
  for (MachineInstr *MI : PendingSpills) {
    SpillBlocks.insert(MI->getParent());
  }

  // 2. 利用机器支配树 (MachineDominatorTree) 寻找安全放置点:
  // 若多条分支共享同一处必须溢出的路径，将其合并提升至这些分支的最近公共支配块 (NCD)，避免重复存储；
  // 反之，若主干路径无需溢出，则将 Store 下沉 (Sink) 到冷分支入口，使热路径实现 0 内存写开销！
  MachineBasicBlock *OptimalDomBB = findNearestCommonDominator(SpillBlocks, MDT);
  if (isProfitableToSinkIntoColdBranches(OptimalDomBB, SpillBlocks, MBFI)) {
    emitSpillsInIndividualColdBlocks(SpillBlocks);
  } else {
    emitSingleMergedSpillAt(OptimalDomBB);
  }
}
```

* **逐行机械执行分析**：

  1. **热路径零溢出保护（Hot-Path Spill Avoidance）**：在 GPU 算子（例如带有罕见边界越界回退或异常日志的内核）中，寄存器压力峰值往往出现在冷路径或边界处理块中。`InlineSpiller` 结合 `MachineBlockFrequencyInfo`（基本块执行频率），精准地将 `BUFFER_STORE_DWORD`（Scratch 溢出写）下沉到低概率冷块内部，确保高频主循环路径既不消耗额外的内存带宽，也不产生 `vmcnt` 等待气泡。

  2. **重物质化链式溯源（`traceSiblingValue`）**：在经历多次区间切分（Live Range Splitting）后，当前被选为溢出牺牲品的虚拟寄存器 `%vreg_split2` 可能只是 `%vreg_split1` 的一个拷贝，而 `%vreg_split1` 又是最初廉价常量加载 `%vreg_orig = S_MOV_B32 0x3f800000` 的副本。`InlineSpiller` 调用 `traceSiblingValue` 逆着 `COPY` 链与 `PHI` 节点一路向上穿透溯源，最终锁定最源头的 `S_MOV_B32`，在消费点直接原地重物质化出常量，彻底省去 Scratch 栈分配。

---

### 3. 【NVPTX 常量折叠与位域/字节置换指令选择】`NVPTXLowerUnreachable.cpp` 与 `NVPTXISelLowering.cpp` 中的 `trap` / `unreachable` 硬件异常降级及 SIMT 死锁规避

* **题目描述**：在 CUDA C++ 核函数中，当发生断言失败（`assert(false)`）、越界触发 `__builtin_trap()` 或编译器推导出的不可达路径（LLVM IR `unreachable` 指令）时，后端必须将其降级为合法的 PTX 终止或异常指令。在单线程 CPU 上，`unreachable` 可以直接忽略或生成 `ud2` 陷阱指令；但在 SIMT 架构的 GPU 上，若某个线程触发陷阱，硬件的异常处理行为（全局上下文销毁 vs 单 Warp 挂起）会直接影响同 SM 上其他并发线程块的执行与调试体验。请结合 `llvm/lib/Target/NVPTX/NVPTXLowerUnreachable.cpp` 与 `llvm/lib/Target/NVPTX/NVPTXIntrinsics.td` 源码，分析后端是如何将 `llvm.trap` 与 `llvm.debugtrap` 分别映射为 PTX `trap;` 与 `brkpt;` 指令的？为什么在某些开启安全检查或高可靠推理模式下，后端选择在 `unreachable` 处发射 `exit;`（仅终止当前线程）而非 `trap;`（触发整个 GPU Context 崩溃报错）？

* **源码文件直达**：

  * `llvm/lib/Target/NVPTX/NVPTXLowerUnreachable.cpp`

  * `llvm/lib/Target/NVPTX/NVPTXIntrinsics.td`

* **核心代码块逻辑**：

```cpp
// llvm/lib/Target/NVPTX/NVPTXLowerUnreachable.cpp 核心逻辑精选
bool NVPTXLowerUnreachable::runOnMachineFunction(MachineFunction &MF) {
  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    // 1. 定位中端遗留的不可达占位指令或内置陷阱调用
    for (MachineInstr &MI : llvm::make_early_inc_range(MBB)) {
      if (MI.getOpcode() == NVPTX::TRAP) {
        // 映射为 PTX 硬件异常陷阱指令 "trap;" -> 触发硬件异常中断并销毁整个 CUDA Context
        Changed = true;
      }
    }

    // 2. 若基本块没有任何后继块且未以合法返回/退出指令结尾 (对应 LLVM IR 的 unreachable):
    // 必须显式插入 "exit;" (终止当前线程执行)，绝不能留空让硬件 PC 越界滑入相邻函数指令区！
    if (MBB.succ_empty() && (MBB.empty() || !MBB.back().isTerminator())) {
      if (TrapUnreachable) {
        BuildMI(MBB, MBB.end(), DebugLoc(), TII->get(NVPTX::TRAP));
      } else {
        // 默认安全模式: 发射线程级退出指令 "exit;"，仅让当前异常线程退出调度，不炸毁同卡其他正常任务！
        BuildMI(MBB, MBB.end(), DebugLoc(), TII->get(NVPTX::EXIT));
      }
      Changed = true;
    }
  }
  return Changed;
}
```

* **逐行机械执行分析**：

  1. **为什么空 `unreachable` 块在 GPU 上会导致灾难？**：在物理指令存储器中，一个 Kernel 的二进制机器码紧挨着下一个 Kernel 的机器码顺序排布。如果编译期由于静态推导认为某分支不可达而直接删除了跳转指令且未放置终结符，一旦运行时因为未定义行为（UB）意外滑入该基本块，SM 的程序计数器（PC）会直接顺延执行下一个无关函数的汇编指令！插入 `exit;` 明确指示 Warp Scheduler 将该线程的活跃掩码（Active Mask）永久置 0 并释放硬件资源。

  2. **`trap;` vs `exit;` 的工程隔离抉择**：`trap;` 会向主机端驱动抛出不可恢复的 `cudaErrorIllegalInstruction` 硬件异常，导致多租户推理服务器上共享同一张 GPU 的所有正常推理请求全部崩溃中断；而 `exit;` 仅静默终止触发异常的当前线程，在生产环境的容错服务中提供了关键的隔离保护。

---

### 4. 【MLIR 向量分发与线程网格映射】`GPU/Transforms/KernelOutlining.cpp` 中的主机-设备（Host-Device）边界切分与隐式捕获变量闭包提升

* **题目描述**：在 MLIR 异构编译流水线（如 IREE、Triton 或 MLIR-GPU 管线）中，高层代码通常在同一个函数体内混合编写主机端循环（Host Control Flow）与 `gpu.launch` 设备端并行核函数体（Device Region）。在降级到 LLVM/NVVM/ROCDL 之前，编译器必须将 `gpu.launch` 的区域剥离出来，封装为独立的 `gpu.module` 与 `gpu.func` 内核函数。请结合 `mlir/lib/Dialect/GPU/Transforms/KernelOutlining.cpp` 源码，深度剖析 `outlineKernelFuncImpl` 是如何扫描 `gpu.launch` 内部所有引用外部 SSA 值的操作数（Implicit Captures，即来自主机侧的标量维度、张量基指针与跨步步长），自动构建新的内核函数参数列表，并在主机侧原位替换为 `gpu.launch_func` 异步启动调用的？在遇到外部捕获值为常量（`arith.constant`）时，Outliner 为何选择将其直接克隆内联进设备内核体中，而非作为内核参数通过 Constant Bank 传递？

* **源码文件直达**：

  * `mlir/lib/Dialect/GPU/Transforms/KernelOutlining.cpp`

* **核心代码块逻辑**：

```cpp
// mlir/lib/Dialect/GPU/Transforms/KernelOutlining.cpp 核心逻辑精选
gpu::GPUFuncOp outlineKernelFuncImpl(gpu::LaunchOp launchOp, StringRef kernelFnName) {
  // 1. 收集所有定义在 gpu.launch 外部 (主机侧)、但在内部被使用的 SSA 值 (Implicit Captures)
  llvm::SetVector<Value> operands;
  getUsedValuesDefinedAbove(launchOp.getBody(), operands);

  // 2. 常量克隆优化 (Constant Cloning):
  // 若某个外部捕获值是由 arith.constant 或 gpu.subgraph 常量生成的标量/向量常量:
  // 绝不将其作为内核参数通过 Kernel Launch Argument Buffer 传递！
  // 直接克隆一份常量指令塞进设备内核入口块，节省内核参数槽位与运行时 Constant Bank 加载指令！
  SmallVector<Value> kernelArgs;
  for (Value v : operands) {
    if (Operation *defOp = v.getDefiningOp(); defOp && defOp->hasTrait<OpTrait::ConstantLike>()) {
      cloneConstantIntoKernelBody(defOp, launchOp.getBody());
    } else {
      kernelArgs.push_back(v); // 真正的运行时动态变量 (如设备显存基指针、动态维度 N) 加入参数列表
    }
  }

  // 3. 构建新的 gpu.func 设备内核签名 (包含显式传递的动态参数)，
  // 将原 region 完整迁移进去，并在主机侧原位替换为异步启动算子 gpu.launch_func！
  return buildOutlinedGPUFuncAndReplaceLaunch(launchOp, kernelFnName, kernelArgs);
}
```

* **逐行机械执行分析**：

  1. **自动化闭包转换（Closure Conversion for Heterogeneous Execution）**：在 CPU-GPU 混合编程模型中，设备端无法直接读取主机端的栈变量寄存器。`getUsedValuesDefinedAbove` 精确捕获所有跨越主机-设备边界的 SSA 依赖（如 `memref` 描述符拆解后的基指针与步长），自动打包成内核启动参数结构体，在主机侧通过驱动 API（`cuLaunchKernel` / `hipModuleLaunchKernel`）拷贝至设备 Constant Bank。

  2. **常量克隆消除参数膨胀**：高层 MLIR 经过折叠后，主机侧往往产生大量局部常量（如分块步长 `128`、零填充常量 `0.0f`）。如果全部作为内核参数传递，不仅很快耗尽硬件固定的 4KB 内核参数空间上限，还迫使设备端在入口发射大量 `ld.param` 指令。通过就地克隆常量进设备函数，这些值在后端直接折叠为算术指令的立即数操作数（Immediate Operands），实现零参数带宽开销。

---

### 5. 【AI 编译器全图算子融合与反向传播重计算】FP8 延迟缩放（Delayed Scaling）与块级动态量化（Per-Block / Per-Token Dynamic Quantization）中的全网最大值（Amax）归约融合优化

* **题目描述**：在基于 Transformer Engine / Cutlass / Triton 的 FP8 大模型训练与推理管线中，由于 FP8（尤其是 `E4M3` 格式，最大动态范围仅为 $\pm 448$）极易发生上溢（Overflow）或下溢归零，每个线性层（GEMM）前后都必须执行严格的 **动态缩放因子计算（`scale = 448.0 / max(abs(X))`）与类型转换量化（`X_fp8 = cast_fp8(X * scale)`）**。如果将“求 `absmax` $\to$ 算 `scale` $\to$ 逐元素量化 $\to$ 写入 HBM”拆分为多个独立算子，会产生多次全张量全局显存读写往返，彻底抵消 FP8 带来的带宽红利。请结合现代 AI 编译器（Triton / Cutlass Epilogue Fusion）架构分析：编译器是如何通过“单内核两遍扫描（Single-Kernel Two-Pass Register/SRAM Resident Quantization）”或“前序 GEMM Epilogue 融合（Fused Amax & Quantize Epilogue）”将全局绝对值最大值归约（`atomicMax` / Block Reduce）与 FP8 量化写回合并在同一个内核中的？在跨 Thread Block 聚合全局 `amax` 时，编译器如何利用浮点单调递增的 IEEE-754 位模式特性（对非负浮点数直接使用无符号整型原子最大值 `atom.global.max.u32`）避免昂贵的软件 CAS 浮点原子循环？

* **核心架构直达**：

  * NVIDIA Transformer Engine / Cutlass 3.x FP8 Epilogue (`epilogue::thread::LinearCombinationGeneric` + `AmaxReduction`)

  * Triton / PyTorch Inductor `fused_fp8_quantize` 内核架构

* **Fused GEMM Epilogue 同步计算 Amax 与 FP8 量化核心伪代码**：

```cpp
// 典型的 Fused GEMM Epilogue: 寄存器内原地计算块级 Amax 并执行无锁整型原子最大值归约
void emitFusedFp8AmaxAndQuantizeEpilogue(OpBuilder &b, Location loc,
                                         Value fp32AccRegs,      // Tensor Core 输出的 FP32 累加器片段
                                         Value globalScalePtr,   // 当前张量的预设/动态缩放因子指针
                                         Value globalAmaxPtr,    // 全局显存中的单标量 Amax 输出地址
                                         Value fp8OutGlobalPtr) {
  // 1. 加载全局缩放因子 scale (标量广播加载，全 Block 共享)
  Value scale = emitUniformScalarLoad(b, loc, globalScalePtr);

  // 2. 在通用寄存器中遍历累加器片段:
  // (a) 同步跟踪局部线程绝对值最大值 localThreadMax = max(localThreadMax, abs(val))
  // (b) 乘以缩放因子并转换为硬件 FP8 (E4M3) 格式: q_val = cvt.rn.satfinite.e4m3x2.f32(val * scale)
  Value localThreadMax = zeroFloatConst;
  SmallVector<Value> packedFp8Regs;
  for (Value val : unpackFp32AccRegs(fp32AccRegs)) {
    Value absVal = b.create<math::AbsFOp>(loc, val);
    localThreadMax = b.create<arith::MaximumFOp>(loc, localThreadMax, absVal);

    Value scaledVal = b.create<arith::MulFOp>(loc, val, scale);
    packedFp8Regs.push_back(emitHardwareCastToFp8(b, loc, scaledVal));
  }

  // 3. 写入全局 FP8 输出矩阵 (128-bit 宽向量合并写回，带宽相比 FP16 减半！)
  emitCoalescedVectorStore128(b, loc, fp8OutGlobalPtr, packedFp8Regs);

  // 4. 多级归约全局 Amax:
  // 第一级: Warp 内寄存器洗牌归约 (shfl.sync.bfly) -> 得到单 Warp 最大值 warpMax
  // 第二级: Shared Memory 跨 Warp 归约 -> 由 Thread 0 得到整个 CTA 的块最大值 blockMax
  Value blockMax = emitBlockReduceMax(b, loc, localThreadMax);

  // 5. 核心硬件黑魔法: 利用非负 IEEE-754 浮点数的严格单调位序特性，
  // 直接将 FP32 位模式 Bitcast 为 uint32_t，发射单条无锁 L2 硬件原子最大值指令:
  // atom.global.max.u32 [%globalAmaxPtr], %blockMaxBits;
  // 彻底省去昂贵的 atomicCAS 循环与 SM 级自旋锁争用！
  b.create<scf::IfOp>(loc, isThreadZero(b, loc), [&](OpBuilder &b, Location loc) {
    Value blockMaxU32 = b.create<arith::BitcastOp>(loc, b.getI32Type(), blockMax);
    emitHardwareAtomicMaxU32(b, loc, globalAmaxPtr, blockMaxU32);
    b.create<scf::YieldOp>(loc);
  });
}
```

* **逐行机械执行分析**：

  1. **IEEE-754 非负浮点位序同构性（Bitwise Order Isomorphism）**：对于所有非负浮点数（符号位 `bit[31] == 0`，包括 $+0.0$、非规格化数、规格化数直至 $+\infty$），其在 IEEE-754 二进制表示下的指数位（`bits[30:23]`）和高位尾数（`bits[22:0]`）的字典序，与 32 位无符号整数（`uint32_t`）的数值大小顺序**100% 严格一致**。由于 `abs(val)` 保证了数值恒非负，编译器直接将其 bitcast 为 `uint32_t` 调用硬件原生 `atom.global.max.u32`（AMD 上对应 `GLOBAL_ATOMIC_UMAX`），在 L2 内存控制器内单周期完成全芯片数千个 CTA 的全局最大值聚合。

  2. **三合一融合消除带宽墙**：如果不做融合，前向 `GEMM` 写回 FP16 结果（1 遍 DRAM 写），`amax` 算子读取 FP16 算最大值（1 遍 DRAM 读），`quantize` 算子再次读取 FP16 并写回 FP8（1 遍 DRAM 读 + 1 遍 DRAM 写）。通过将 `amax` 统计与 `cast_fp8` 全部直接融合进上游 `GEMM` 的 Epilogue 寄存器流水线中，中间的 FP16 结果从头到尾没有触碰过一次全局显存，端到端层间延迟直接降低 35%\~50%。
