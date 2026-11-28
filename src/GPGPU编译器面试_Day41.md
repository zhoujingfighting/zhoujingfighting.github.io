# GPGPU 编译器模拟面试与深度解析 —— Day 41：统一寄存器池 AGPR 重写、`MachineLICM` 压力感知、Hopper TMA Reduce 与 Ring Attention

### 1. 【AMDGPU 矩阵核心与统一寄存器池】`AMDGPURewriteAGPRCopyMFMA.cpp` 中的 AGPR 物理寄存器重分配与零拷贝累加

* **题目描述**：在 AMD CDNA2/CDNA3（MI250X / MI300X）微架构中，寄存器文件引入了统一寄存器池（Unified Register File），允许常规指令与 MFMA 指令直接读写累加器向量寄存器（AGPR，即 `a[0:255]`），消除了初代 CDNA 架构中必须在 VGPR 与 AGPR 之间通过 `v_accvgpr_read` / `v_accvgpr_write` 来回拷贝的开销。请结合 `llvm/lib/Target/AMDGPU/AMDGPURewriteAGPRCopyMFMA.cpp` 源码，深度剖析编译器在寄存器分配（`RegAllocGreedy`）之后是如何识别围绕 `MFMA` 指令的 `COPY` 链，并将原本分配在 VGPR 上的输入/输出操作数原地重命名为 AGPR（替换对应的 MFMA 寄存器类约束）的？在什么情况下跨基本块或带子寄存器约束的 `COPY` 无法被消除？

* **源码文件直达**：

  * `llvm/lib/Target/AMDGPU/AMDGPURewriteAGPRCopyMFMA.cpp`

  * `llvm/lib/Target/AMDGPU/SIRegisterInfo.cpp`

* **核心代码块逻辑**：

```cpp
// llvm/lib/Target/AMDGPU/AMDGPURewriteAGPRCopyMFMA.cpp 核心逻辑精选
bool AMDGPURewriteAGPRCopyMFMA::runOnMachineFunction(MachineFunction &MF) {
  // 1. 仅在支持统一 AGPR/VGPR 寻址的 CDNA2+ (GFX90A / GFX940+) 架构上启用
  if (!ST->hasGFX90AInsts())
    return false;

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      // 2. 定位矩阵乘加指令 (如 V_MFMA_F32_16X16X16_F16_vgprcd_e64)
      // 在寄存器分配期间，为避免过早锁死稀缺的 AGPR，调度器可能默认选择了 VGPR 输出变体
      if (!TII->isMFMA(MI))
        continue;

      // 3. 检查 MFMA 的累加器输入 (Src2) 或输出 (Dst) 是否被冗余的 COPY 包裹:
      // 例如: %agpr0 = COPY %vgpr0; %agpr1 = V_MFMA(..., %agpr0); %vgpr1 = COPY %agpr1
      // 查找对应的纯 AGPR 编码指令变体 (如 V_MFMA_F32_16X16X16_F16_e64，直接读写 a[0:3])
      int AGPROpcode = AMDGPU::getMFMASrcCVGPRToAGPROpcode(MI.getOpcode());
      if (AGPROpcode == -1)
        continue;

      // 4. 验证目标 VGPR 对应的物理编号在 AGPR 文件中完全空闲且无活跃冲突 (LivePhysRegs)
      if (canReplaceVGPRWithAGPR(MI, AGPROpcode)) {
        MI.setDesc(TII->get(AGPROpcode));
        rewriteOperandsAndEraseCopies(MI);
        Changed = true;
      }
    }
  }
  return Changed;
}
```

* **逐行机械执行分析**：

  1. **GFX90A 统一寄存器池微架构演进**：在初代 CDNA1（MI100）中，VGPR 和 AGPR 是物理分离的独立 SRAM，将数据喂给 MFMA 必须先发射 `v_accvgpr_write_b32`，算完再发射 `v_accvgpr_read_b32`，不仅耗费双倍指令，还因为写后读硬件气泡拖垮矩阵流水线。从 MI200（GFX90A）开始，硬件将 512 个向量寄存器统一为一个物理池，软件通过内核元数据 `amdhsa_accum_offset` 静态划分 VGPR 与 AGPR 分界线，允许 `V_MFMA`、`BUFFER_LOAD` 和 `V_ADD` 直接操作 AGPR。

  2. **Post-RA 原地指令变体升级（Opcode Morphing）**：在寄存器分配阶段，分配器为了防止过早分配导致 AGPR 碎片化，可能先将数据留在 VGPR 上；`AMDGPURewriteAGPRCopyMFMA` 在分配完成后扫描指令流，一旦确认目标物理槽位在 AGPR 空间可用且满足对齐要求，直接篡改 `MachineInstr` 的 `MCInstrDesc` 操作码，将 VGPR 变体升级为 AGPR 变体并抹除全部搬运指令。若 `COPY` 跨越了存在其他消费者或寄存器类别不兼容（如部分硬件指令不支持 AGPR 编码）的指令边界，则保守保留原搬运指令。

---

### 2. 【LLVM 机器级循环外提与寄存器压力悬崖】`MachineLICM.cpp` 中的 GPU 寄存器压力感知约束与净压力降定律

* **题目描述**：在通用 CPU 编译优化中，机器级循环不变代码外提（`MachineLICM`）倾向于将循环体内所有不依赖循环归纳变量的计算全部外提到循环前置块（Loop Preheader）中。然而在 GPGPU 架构中，激进的 `MachineLICM` 会将大量短命值变成横跨整个循环的长生命周期变量，引爆 VGPR 寄存器压力并导致 SM Occupancy 暴跌。请结合 `llvm/lib/CodeGen/MachineLICM.cpp` 源码，深度剖析 `MachineLICMBase::IsProfitableToHoist` 是如何通过 `TargetRegisterInfo::getRegPressureSetLimit` 与活跃物理寄存器压降估算（Register Pressure Cost Model）抑制过度外提的？在什么情况下外提一条指令反而能“降低”循环体内的寄存器压力？

* **源码文件直达**：

  * `llvm/lib/CodeGen/MachineLICM.cpp`

  * `llvm/lib/Target/AMDGPU/SIInstrInfo.cpp`

* **核心代码块逻辑**：

```cpp
// llvm/lib/CodeGen/MachineLICM.cpp 核心逻辑精选
bool MachineLICMBase::IsProfitableToHoist(MachineInstr &MI) {
  // 1. 若指令本身具备低成本重物质化属性 (isAsCheapAsAMove，例如 S_MOV_B32 加载立即数)，
  // 在高频循环内部直接就地生成比在循环外占用一个物理寄存器一路带进来更划算！
  if (TII->isAsCheapAsAMove(MI) && !isHighLatencyLoad(MI)) {
    return false; // 拒绝外提廉价常量指令，保护宝贵的 GPU 寄存器配额
  }

  // 2. 计算外提该指令对循环内寄存器压力的净影响 (Net Register Pressure Delta):
  // Delta = (MI 定义的寄存器集合权重) - (因外提而在循环内彻底死亡的源操作数权重)
  // 经典逆向直观场景: 若循环内有一条指令 %dst = op %in1, %in2，其中 %in1 和 %in2 都是循环不变的
  // (各自占 1 个 VGPR)，且它们在循环内没有其他消费者:
  // 外提之前: %in1 和 %in2 必须全程存活贯穿整个循环，占用 2 个 VGPR！
  // 外提之后: 计算提前在 Preheader 完成，%in1 和 %in2 在进入循环前就已死亡，循环内只需携带 1 个 %dst，
  // 寄存器占用直接从 2 个降为 1 个 (Net Delta = -1)！
  if (reducesLoopRegisterPressure(MI)) {
    return true;
  }

  // 3. 若外提会增加循环内的活跃寄存器数量，查询目标架构当前压力集合上限:
  // 若当前循环的 VGPR/SGPR 压力已逼近 Occupancy 悬崖阈值 (如 64 或 128 个 VGPR)，立即否决外提！
  return !willExceedTargetOccupancyThreshold(MI, CurLoop);
}
```

* **逐行机械执行分析**：

  1. **寄存器压力净减少（Net Pressure Reduction）定律**：当一条指令消费 $K$ 个仅在此处使用的循环不变输入并产生 $1$ 个输出（且 $K > 1$）时，外提该指令会让 $K$ 个输入变量在循环开始前就结束生命周期，循环内部只需维持 $1$ 个输出变量，净节省 $K - 1$ 个物理寄存器。`MachineLICM` 通过精确追踪源操作数的 `isKill` 属性捕捉这一红利。

  2. **Occupancy 悬崖（Occupancy Cliff）精确卡位**：在 GPU 上，VGPR 数量往往以阶梯状（如 32、64、96、128）决定每个 SM 能驻留的 Wave 数量。若一个内核原本消耗 63 个 VGPR（每 SM 可跑 8 个 Wave），盲目外提 2 个地址计算变量使其变成 65 个 VGPR，会导致并发 Wave 数瞬间腰斩至 4 个，算力吞吐暴跌 50%。`SIInstrInfo` 通过重写寄存器压力限制钩子，在接近阶梯边缘时果断锁死 `MachineLICM` 外提。

---

### 3. 【NVPTX 张量存储与硬件归约卸载】`NVPTXIntrinsics.td` 中的 Hopper `cp.reduce.async.bulk.tensor`（TMA Reduce）与跨代理可见性闭环

* **题目描述**：在 NVIDIA Hopper（SM_90）架构中，TMA（Tensor Memory Accelerator）不仅支持异步加载与存储，还支持 **硬件级带归约的异步张量存储（TMA Reduce，如 `cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.f16`）**，允许线程块直接将 Shared Memory 输出 Tile 异步推送到全局显存并由 L2 内存控制器原地完成浮点累加。请结合 `llvm/lib/Target/NVPTX/NVPTXIntrinsics.td` 与 `llvm/lib/Target/NVPTX/NVPTXISelLowering.cpp` 源码，分析后端是如何降级 `@llvm.nvvm.cp.reduce.async.bulk.tensor` 内建函数的？在启动 TMA Reduce 之前，为什么编译器必须插入 `fence.proxy.async.shared::cta` 与 `cp.async.bulk.commit_group` 形成内存可见性闭环？

* **源码文件直达**：

  * `llvm/lib/Target/NVPTX/NVPTXIntrinsics.td`

  * `llvm/lib/Target/NVPTX/NVPTXISelLowering.cpp`

* **核心模式定义**：

```
// llvm/lib/Target/NVPTX/NVPTXIntrinsics.td 核心模式定义精选
// Hopper (SM_90+) TMA 异步张量归约存储指令:
// 将 Shared Memory ($src) 中的二维矩阵瓦片异步推送到全局显存 ($tmap 描述符指定的坐标 {$x, $y})，
// 并由 L2 内存控制器在落盘瞬间执行硬件原子浮点累加 (.add.f16 / .add.bf16 / .add.f32)！
let hasSideEffects = 1 in {
  def INT_NVVM_CP_REDUCE_ASYNC_BULK_TENSOR_S2G_2D_ADD_F16 :
      NVPTXInst<(outs),
                (ins Int64Regs:$tmap, Int32Regs:$x, Int32Regs:$y, Int32Regs:$src),
                "cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group.f16 "
                "[$tmap, {$x, $y}], [$src];",
                [(int_nvvm_cp_reduce_async_bulk_tensor_s2g_2d_add_f16
                    addr:$tmap, i32:$x, i32:$y, addr:$src)]>;
}
```

* **逐行机械执行分析**：

  1. **告别线程级 Epilogue 写回瓶颈**：在传统 GEMM Epilogue 中，每个线程必须计算出自己的全局行列坐标，做越界条件判断，再发射数十条 `red.global.v4.f16` 指令。通过 TMA Reduce，整个 Warpgroup 先将累加器结果整整齐齐写入 Shared Memory 缓冲区（100% 无 Bank Conflict），随后仅由单个选举线程发射一条 `cp.reduce.async.bulk.tensor`，硬件 TMA 引擎自动按多维张量步长拆分突发事务并推送至 L2 控制器执行原地归约，释放出全部 SM 算力。

  2. **跨代理可见性闭环（`fence.proxy.async` + `bulk_group`）**：由于写入 Shared Memory 的动作由常规线程（Generic Proxy）执行，而读取 Shared Memory 并向外发送的动作由 TMA 引擎（Async Proxy）执行，编译器必须在 `st.shared` 与 TMA Reduce 之间插入 `fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster` 刷新写缓冲区；在发射 TMA Reduce 后，则通过 `cp.async.bulk.commit_group` 和 `cp.async.bulk.wait_group.read 0` 确保 Shared Memory 缓冲区在被下一次迭代覆写前已被 TMA 引擎完全读取完毕。

---

### 4. 【MLIR 内存视图折叠与线性化消除】`FoldMemRefAliasOps.cpp` 中的嵌套 `memref.subview` 索引复合与描述符削减

* **题目描述**：在 AI 编译器中，经过高层分块（Tiling）与 Bufferization 后，生成的 GPU 内核代码中充斥着大量嵌套的 `memref.subview` 视图切片操作。如果在降级到 LLVM IR 前未能将嵌套的 `subview` 折叠消除，每个 `subview` 都会在 IR 中维持一个包含基指针、对齐指针、偏移量、多维尺寸和多维跨步（Strides）的庞大结构体描述符，在 GPU 后端引发严重的寄存器膨胀与整数乘法地址计算开销。请结合 `mlir/lib/Dialect/MemRef/Transforms/FoldMemRefAliasOps.cpp` 源码，深度剖析 `LoadOpOfSubViewOpFolder` 是如何利用仿射映射代数（`resolveSourceIndices`）将多层嵌套的 `subview` 偏移与步长直接复合进底层原始 `memref` 的多维下标表达式中的？

* **源码文件直达**：

  * `mlir/lib/Dialect/MemRef/Transforms/FoldMemRefAliasOps.cpp`

  * `mlir/lib/Dialect/MemRef/IR/MemRefOps.cpp`

* **核心代码块逻辑**：

```cpp
// mlir/lib/Dialect/MemRef/Transforms/FoldMemRefAliasOps.cpp 核心逻辑精选
LogicalResult LoadOpOfSubViewOpFolder<OpTy>::matchAndRewrite(
    OpTy loadOp, PatternRewriter &rewriter) const {
  // 1. 识别访存操作的目标对象是否为 memref.subview 产生的视图切片
  auto subViewOp = loadOp.getMemRef().template getDefiningOp<memref::SubViewOp>();
  if (!subViewOp)
    return failure();

  // 2. 利用仿射代数合成新索引列表:
  // 对于第 d 维: orig_index[d] = subview_offset[d] + load_index[d] * subview_stride[d]
  // 若存在维数降秩 (Rank-Reducing SubView，例如从 3D 张量切出 2D 切片)，
  // 自动在被消去的维度位置填入对应的静态常量偏移量！
  SmallVector<Value> indices;
  if (failed(resolveSourceIndices(loadOp.getLoc(), rewriter, subViewOp,
                                  loadOp.getIndices(), indices)))
    return failure();

  // 3. 原地替换访存目标为原始父级 MemRef (subViewOp.getSource())，彻底摆脱中间子视图描述符依赖！
  rewriter.replaceOpWithNewOp<OpTy>(loadOp, subViewOp.getSource(), indices);
  return success();
}
```

* **逐行机械执行分析**：

  1. **消除 MemRef Descriptor 结构体拆分开销**：在 MLIR 降级至 LLVM IR 时，一个 $N$ 维 `memref` 会被展开为包含 $2N + 3$ 个标量字段的 LLVM Struct（对于 4D MemRef 高达 11 个 64 位寄存器！）。`FoldMemRefAliasOps` 将所有嵌套子视图层层剥离，使所有访存直接指向最顶层的根 MemRef，配合下游的 `Canonicalizer` 与 `CSE`（公共子表达式消除），将多维偏移量折叠为单一常数立即数，释放数十个寄存器。

  2. **解锁下游向量化与共享内存 Bank 冲突分析**：许多 GPU 专属分析 Pass（如对齐推导与共享内存 Swizzle 注入）要求访存操作直接作用于分配根源（`memref.alloc`）。通过折叠掉中间 `subview`，分析器得以直接追踪到共享内存分配基址，精确计算出跨线程的物理 Bank 映射。

---

### 5. 【AI 编译器分布式长上下文流水线】Ring Attention 跨卡异步通信与计算重叠（Comm-Compute Overlap）及 Striped Attention 负载均衡

* **题目描述**：在训练或推理超长上下文大模型（如 1M \~ 10M Tokens 上下文窗口）时，工业界采用 **Ring Attention（Context Parallelism）** 将序列切分到 $P$ 张 GPU 上，各卡在环形拓扑中一边计算当前本地 KV 分块的 FlashAttention，一边通过 NVLink 异步向下一张卡发送/接收下一个 KV 分块。请结合现代分布式 AI 编译器（如 XLA GSPMD / PyTorch Compile + Async Collective）架构分析：编译器是如何通过将通信原语切分为细粒度微块，并利用双缓冲软件流水线调度将其完全隐藏在本地 FlashAttention 计算阴影下的？面对因果掩码（Causal Mask）导致的跨卡负载严重倾斜，编译器如何通过 Striped Attention 拓扑重排实现全卡算力均摊？

* **核心架构直达**：

  * XLA `CollectivePermuteDecomposer` & `AsyncCollectiveCreator`

  * Ring Attention / Striped Attention 分布式编译器流水线模型

* **Ring Attention 计算与 P2P 异步通信双缓冲重叠核心伪代码**：

```cpp
// 典型的 Ring Attention 分布式计算-通信双缓冲软件流水线调度模型
void emitRingAttentionPipelinedLoop(OpBuilder &b, Location loc,
                                    Value qLocal, Value kInitLocal, Value vInitLocal,
                                    int ringSize, int myRank) {
  // 1. 分配双缓冲片上显存槽位 (Ping-Pong HBM Buffers):
  // kBuffers[0], vBuffers[0] 存放当前步正在参与 Tensor Core 计算的 KV 块
  // kBuffers[1], vBuffers[1] 存放后台 NVLink DMA 引擎正在从左邻居接收的下一步 KV 块
  Value kBuf[2] = {kInitLocal, allocateHbmBufferLike(b, loc, kInitLocal)};
  Value vBuf[2] = {vInitLocal, allocateHbmBufferLike(b, loc, vInitLocal)};

  Value accOut = zeroInitOut(b, loc, qLocal);
  Value globalMax = negInfInit(b, loc, qLocal);
  Value globalSum = zeroInitSum(b, loc, qLocal);

  // 2. 启动环形步进循环 (共 ringSize 步):
  for (int step = 0; step < ringSize; ++step) {
    int currSlot = step % 2;
    int nextSlot = (step + 1) % 2;

    // 3. 异步发射下一轮的点对点环形通信 (Async CollectivePermute / NCCL SendRecv):
    // 除最后一步外，立即在专用通信流 (Comm Stream) 上启动异步发送与接收，不阻塞当前 SM 计算！
    AsyncToken commToken;
    if (step + 1 < ringSize) {
      int sendTargetRank = (myRank + 1) % ringSize;
      int recvSourceRank = (myRank - 1 + ringSize) % ringSize;
      commToken = emitAsyncCollectivePermute(
          b, loc, /*sendData=*/{kBuf[currSlot], vBuf[currSlot]},
          /*recvDest=*/{kBuf[nextSlot], vBuf[nextSlot]}, sendTargetRank, recvSourceRank);
    }

    // 4. 在主计算流 (Compute Stream) 上全速执行当前本地分块的 FlashAttention 内核！
    // 注意: 若启用 Causal Mask (因果掩码)，编译器根据 (myRank, sourceBlockRank) 静态裁剪掉
    // 未来 Token 块 (Skipped Future Blocks)，实现零算力浪费！
    int sourceBlockRank = (myRank - step + ringSize) % ringSize;
    if (!isStrictlyFutureCausalBlock(myRank, sourceBlockRank)) {
      auto [blockOut, blockLse] = emitFusedFlashAttentionBlock(
          b, loc, qLocal, kBuf[currSlot], vBuf[currSlot],
          /*applyCausalMask=*/(myRank == sourceBlockRank));

      // 5. 增量 Online Softmax 合并: 利用 Log-Sum-Exp 重缩放因子合并跨卡分块结果
      emitIncrementalOnlineSoftmaxUpdate(b, loc, accOut, globalMax, globalSum, blockOut, blockLse);
    }

    // 6. 同步等待异步通信令牌完成，确保下一轮所需的 kBuf[nextSlot] 已完整抵达显存
    if (step + 1 < ringSize) {
      emitAsyncWait(b, loc, commToken);
    }
  }
}
```

* **逐行机械执行分析**：

  1. **计算-通信时间比（Compute-to-Communication Ratio）掩盖条件**：假设序列分块大小为 $B_c$，隐藏层维度为 $D$。单步 FlashAttention 的浮点计算量约为 $4 B_c^2 D$（FP16），而单步通过 NVLink 发送 $K$ 和 $V$ 的总数据量为 $4 B_c D$ 字节。计算与通信之比为 $\frac{4 B_c^2 D}{4 B_c D} = B_c$。只要每个设备分配的局部序列长度 $B_c > \frac{\text{SM TFLOPS}}{\text{NVLink GB/s}}$（例如在 H100 上 $989 / 900 \approx 1.1\text{K}$，即 $B_c \ge 2048$ Tokens），NVLink 传输延迟就会被 **100% 完美掩盖在 Tensor Core 计算阴影之下**，实现近乎线性的多卡长上下文扩展效率。

  2. **Striped Attention（条纹注意力）解决 Causal Mask 负载倾斜**：在标准 Ring Attention 中，由于因果掩码（Causal Mask）的存在，Rank 0 在第 0 步算完后，后续 $P-1$ 步全部处于未来块跳过状态（算力完全闲置），而 Rank $P-1$ 却要满载计算全部 $P$ 步，造成严重的跨卡木桶效应。先进 AI 编译器通过 **Striped Attention** 重排 Token 分布（将序列按跨步交错分配给各卡，使每张卡同时持有早期与晚期 Token），使所有 GPU 在每一步环形迭代中均承担完全均等的三角形计算负载，彻底消除跨卡同步气泡。
