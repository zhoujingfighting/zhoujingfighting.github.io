# GPGPU 编译器模拟面试与深度解析 —— Day 37：全四元组模式（WQM）、VGPR Bank Conflict、`PRMT` 字节重组、`ldmatrix` 与 PagedAttention

### 1. 【AMDGPU 标量状态与硬件资源描述符】`SIWholeQuadMode.cpp`（WQM）中的全四元组执行模式推导与隐式像素/线程激活传播
* **题目描述**：在 GPGPU 通用计算（如稀疏网格邻域差分、光线追踪或图像纹理采样 `IMAGE_SAMPLE`）中，硬件计算梯度导数（`ddx` / `ddy`）要求 2x2 像素/线程四元组（Quad）内的所有 4 个线程必须同时激活。即使其中部分线程在控制流分支中被逻辑屏蔽，硬件也必须临时开启全四元组执行（Whole Quad Mode, WQM）以防计算出脏导数。请结合 `llvm/lib/Target/AMDGPU/SIWholeQuadMode.cpp` 源码，深度剖析 `SIWholeQuadMode` Pass 是如何在数据流图上执行反向（Backward）与前向（Forward）传播，精确界定哪些基本块必须升级为 `StateWQM`、哪些保持 `StateExact` 的？Pass 如何在模式切换边界发射 `s_wqm_b64` 与 `s_and_b64 exec, exec, exact_mask` 指令，避免无效线程产生非预期的内存写入副作用？
* **源码文件直达**：
  * `llvm/lib/Target/AMDGPU/SIWholeQuadMode.cpp`
* **核心代码块逻辑**：

```cpp
// llvm/lib/Target/AMDGPU/SIWholeQuadMode.cpp 核心逻辑精选
void SIWholeQuadMode::propagateBackward(MachineFunction &MF) {
  // 1. 反向数据流传播 (Backward Analysis):
  // 凡是隐式依赖邻域线程数据的指令 (如 IMAGE_SAMPLE 纹理梯度导数计算或 WQM 标记指令)，
  // 将其定义的所有输入源操作数指令递归标记为需要 StateWQM！
  while (!Worklist.empty()) {
    MachineInstr *MI = Worklist.pop_back_val();
    for (const MachineOperand &MO : MI->uses()) {
      if (!MO.isReg() || !MO.getReg().isVirtual()) continue;
      MachineInstr *DefMI = MRI->getVRegDef(MO.getReg());
      // 若前驱计算指令尚未处于 WQM 状态，将其升格为 StateWQM 并推入工作列表
      if (DefMI && !isStrictlyExact(*DefMI)) {
        markInstructionWQM(*DefMI);
        Worklist.push_back(DefMI);
      }
    }
  }
}

void SIWholeQuadMode::emitStateTransition(MachineBasicBlock &MBB,
                                          MachineBasicBlock::iterator MI,
                                          char CurrentState, char TargetState) {
  if (CurrentState == StateExact && TargetState == StateWQM) {
    // 2. 切入 WQM 模式: 保存当前精确活跃掩码，随后调用硬件 S_WQM_B64 将任意含活跃线程的 Quad 全部 4 位拉高！
    BuildMI(MBB, MI, DL, TII->get(AMDGPU::S_MOV_B64), SavedExactReg).addReg(AMDGPU::EXEC);
    BuildMI(MBB, MI, DL, TII->get(AMDGPU::S_WQM_B64), AMDGPU::EXEC).addReg(AMDGPU::EXEC);
  } else if (CurrentState == StateWQM && TargetState == StateExact) {
    // 3. 退回 Exact 模式: 遇到全局内存写入 (VMEM Store) 或原子操作前，必须恢复精确掩码，
    // 严禁被临时唤醒的休眠辅助线程 (Helper Lanes) 向显存写入脏数据！
    BuildMI(MBB, MI, DL, TII->get(AMDGPU::S_AND_B64), AMDGPU::EXEC)
        .addReg(AMDGPU::EXEC).addReg(SavedExactReg);
  }
}
```

* **逐行机械执行分析**：
  1. **硬件 `S_WQM_B64` 组合逻辑**：在 AMD 标量单元中，`s_wqm_b64 exec, exec` 在 1 个时钟周期内对 64 位执行掩码按每 4 位一组（bits `[0:3]`, `[4:7]`...）做位或归约：只要组内有任意 1 个线程处于活跃状态，该组全部 4 个位均被置 1。这使得原本在分支中被屏蔽的相邻线程（Helper Lanes）被临时激活，从而为前驱坐标计算提供合法的邻域数据。
  2. **有副作用指令的 `StateExact` 绝对隔离**：辅助线程仅允许执行纯算术与只读纹理采样。一旦指令流触及全局写入（`BUFFER_STORE`）、LDS 写入或原子操作，`SIWholeQuadMode` 必须在该指令前立即发射 `s_and_b64 exec, exec, SavedExactReg` 剥离辅助线程，防止产生越界内存破坏。

---

### 2. 【LLVM 向量寄存器银行冲突与 Bank 仲裁】`GCNHazardRecognizer.cpp` 中的硬件寄存器存储体冲突（VGPR Bank Conflict）与操作数读取端口调度
* **题目描述**：在现代 GPGPU 微架构（如 AMD GCN/CDNA 与 NVIDIA Volta/Hopper）中，庞大的物理向量寄存器文件（VGPR Register File）并非单端口全局全连接 SRAM，而是被物理切分为多个独立的存储体（Banks，如 4 个或 8 个 Bank，按寄存器编号模 4 映射 `v0 -> Bank 0, v1 -> Bank 1`）。当单条三操作数融合乘加指令（`V_FMA_F32 v4, v0, v8, v12`）在同一个时钟周期试图从同一个 Bank 读取多个不同的物理寄存器时，会发生 **Operand Collector Bank Conflict**，导致指令发射停顿。请结合 `llvm/lib/Target/AMDGPU/GCNHazardRecognizer.cpp` 与 `llvm/lib/CodeGen/RegAllocGreedy.cpp` 源码，分析编译器在指令调度与寄存器分配提示（Allocation Hints）阶段是如何感知物理寄存器 Bank 映射规律并规避同周期 Bank 争用的？
* **源码文件直达**：
  * `llvm/lib/Target/AMDGPU/GCNHazardRecognizer.cpp`
  * `llvm/lib/Target/AMDGPU/SIRegisterInfo.cpp`
* **核心代码块逻辑**：

```cpp
// 典型的 VGPR Bank 冲突评估与调度启发式核心逻辑
unsigned evaluateVGPRBankConflictCost(const MachineInstr &MI,
                                      const TargetRegisterInfo *TRI) {
  // AMD GCN/CDNA 架构将 VGPR 文件划分为 4 个物理 SRAM Bank: BankID = PhysRegNum % 4
  SmallSet<unsigned, 4> UsedBanksThisCycle;
  unsigned ConflictPenalty = 0;

  // 遍历同一周期内并发读取的源操作数 (Src0, Src1, Src2)
  for (unsigned i = 0; i < MI.getNumExplicitOperands(); ++i) {
    const MachineOperand &MO = MI.getOperand(i);
    if (!MO.isReg() || !MO.isUse() || !MO.getReg().isPhysical())
      continue;

    if (SIRegisterInfo::isVGPRPhysReg(MO.getReg())) {
      unsigned PhysIdx = TRI->getEncodingValue(MO.getReg());
      unsigned BankID = PhysIdx & 0x3; // PhysReg % 4

      // 若同一条指令的不同操作数落入相同的物理 Bank (例如 V_FMA_F32 v4, v0, v4, v8 -> 全部命中 Bank 0!)
      // 操作数收集器 (Operand Collector) 无法在单周期读取同一 Bank 的多个地址，被迫插入停顿周期！
      if (!UsedBanksThisCycle.insert(BankID).second) {
        ConflictPenalty += 1; // 每次 Bank 碰撞增加 1 个硬件读取停顿周期
      }
    }
  }
  return ConflictPenalty;
}
```

* **逐行机械执行分析**：
  1. **Operand Collector（操作数收集器）微架构瓶颈**：为了节省芯片面积与功耗，GPU 的物理寄存器文件采用单端口高密度 SRAM Bank 构建（例如 4 个 Bank，每个 Bank 宽度为 32 threads $\times$ 32 bits = 1024 bits）。如果一条 `V_FMA_F32` 的三个源操作数分别分配在 `v0`（Bank 0）、`v4`（Bank 0）和 `v8`（Bank 0），硬件仲裁器需要花费 **3 个连续时钟周期**才能集齐所有操作数，导致 ALU 吞吐骤降至 1/3。
  2. **编译器协同防御机制**：
     * **跨指令交错调度（`MachineScheduler`）**：调度器优先将操作数分布在互异 Bank（如 `v0, v1, v2`）的指令排在相邻发射槽，充分利用总线广播旁路（Operand Forwarding Cache）；
     * **寄存器分配提示（`RegAllocGreedy::getRegAllocationHints`）**：在为三元累加指令分配物理寄存器时，目标钩子主动将与同指令其他源操作数 `PhysReg % 4` 冲突的候选物理寄存器排到优先级列表末尾，诱导着色器优先挑选不同 Bank 的空闲寄存器。

---

### 3. 【NVPTX 常量折叠与位域/字节置换指令选择】`NVPTXISelDAGToDAG.cpp` 中的 `PRMT`（Byte Permute）模式匹配与跨寄存器字节重组
* **题目描述**：在 AI 推理的低比特量化（FP8/INT8 权重交织重组）以及视频编解码内核中，经常需要将两个 32 位寄存器的特定字节抽取拼接为新的 32 位字（例如 `((a & 0xFF00) >> 8) | ((b & 0xFF) << 16)`）。NVIDIA GPU 提供了极其强大的单周期字节混洗指令 `prmt.b32 dst, a, b, selector`。请结合 `llvm/lib/Target/NVPTX/NVPTXISelDAGToDAG.cpp` 与 `llvm/lib/Target/NVPTX/NVPTXInstrInfo.td` 源码，分析指令选择器是如何利用 `SelectionDAG::computeKnownBits` 追踪每个字节槽位的数据来源（Byte Provenance），自动合成 16 进制选择器控制字（`0x76543210` 映射），从而将复杂的移位掩码树坍缩为单条 `PRMT` 指令的？
* **源码文件直达**：
  * `llvm/lib/Target/NVPTX/NVPTXISelDAGToDAG.cpp`
  * `llvm/lib/Target/NVPTX/NVPTXInstrInfo.td`
* **核心代码块逻辑**：

```cpp
// llvm/lib/Target/NVPTX/NVPTXISelDAGToDAG.cpp 核心逻辑精选
bool NVPTXDAGToDAGISel::SelectPRMT(SDNode *N) {
  if (N->getOpcode() != ISD::OR || N->getValueType(0) != MVT::i32)
    return false;

  // 1. 追踪 OR 树叶子节点中所有贡献字节的源寄存器 (最多支持 2 个源 32-bit 寄存器 SrcA, SrcB)
  // 将 SrcA 的 4 个字节编号为 0..3，SrcB 的 4 个字节编号为 4..7
  SDValue SrcA, SrcB;
  unsigned ByteSelector = 0; // 16 位选择器控制字，每 4 位指定输出的一个字节来源 (0..7)

  // 2. 利用 KnownBits 与移位偏移分析，解析输出第 i 个字节 (i = 0..3) 究竟来自 SrcA 或 SrcB 的哪一字节
  for (unsigned ByteIdx = 0; ByteIdx < 4; ++ByteIdx) {
    int SourceByteId = traceByteOrigin(N, ByteIdx, SrcA, SrcB);
    if (SourceByteId < 0)
      return false; // 存在跨字节边界的位操作或未知动态移位，无法用字节级置换表达
    ByteSelector |= (SourceByteId & 0xF) << (ByteIdx * 4);
  }

  // 3. 发射单条硬件字节混洗指令: prmt.b32 dst, SrcA, SrcB, ByteSelector
  SDLoc DL(N);
  SDValue SelConst = CurDAG->getTargetConstant(ByteSelector, DL, MVT::i32);
  ReplaceNode(N, CurDAG->getMachineNode(NVPTX::PRMT_B32_rii, DL, MVT::i32,
                                        SrcA, SrcB, SelConst));
  return true;
}
```

* **逐行机械执行分析**：
  1. **字节溯源（Byte Provenance Tracking）**：以 `(a >> 8) & 0xFF` 为例，输出的第 0 字节（bits `0..7`）明确来自 `a` 的第 1 字节（编号 `1`）；而输出的第 1~3 字节被掩码 `0xFF` 恒定清零（在 PTX `prmt` 模式下，`0x8` 或通过符号扩展位控制零填充）。
  2. **多指令树形坍缩**：在 FP8/INT8 权重解包或矩阵转置中，开发者常写出包含 6~8 条 `shl`、`lshr`、`and`、`or` 的复杂位拼接代码。`SelectPRMT` 直接将其整体折叠为 **1 条 `prmt.b32` 指令**，指令执行周期从 6~8 周期骤降至 **1~2 周期**，同时省去所有中间临时寄存器，极大释放了张量解包循环的寄存器压力。

---

### 4. 【MLIR 分布式张量布局与向量广播降级】`VectorToGPU` 与 `NVGPUToNVVM` 中的 `ldmatrix` 硬件矩阵加载指令降级与地址对齐求解
* **题目描述**：在 NVIDIA Turing / Ampere / Hopper 架构中，`ldmatrix.sync.aligned.m8n8.x4.shared.b16` 允许单条指令让 Warp 内 32 个线程协同从 Shared Memory 一次性加载四个 $8 \times 8$ 的 FP16 矩阵切片到通用寄存器中，且由硬件内部交叉开关自动完成跨线程转置（`.trans` 修饰符）。请结合 `mlir/lib/Conversion/NVGPUToNVVM/NVGPUToNVVM.cpp` 源码，深度剖析 `LdMatrixOpLowering` 是如何将高层 `nvgpu.ldmatrix` 算子映射为 `@llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16` 的？为什么在传入 Shared Memory 指针时，降级逻辑必须先将通用内存指针转换为 32 位 Shared Address Space 偏移（`addrspacecast` to `addrspace(3)`），且不同线程传入的基指针为何只需覆盖矩阵每一行的首地址（仅前 8/16/32 线程有效，其余线程指针被硬件忽略）？
* **源码文件直达**：
  * `mlir/lib/Conversion/NVGPUToNVVM/NVGPUToNVVM.cpp`
* **核心代码块逻辑**：

```cpp
// mlir/lib/Conversion/NVGPUToNVVM/NVGPUToNVVM.cpp 核心逻辑精选
LogicalResult LdMatrixOpLowering::matchAndRewrite(
    nvgpu::LdMatrixOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  Value srcMemref = adaptor.getSrcMemref();
  ValueRange indices = adaptor.getIndices();

  // 1. 计算当前线程在 Shared Memory 缓冲区中的线性字节偏移指针
  Value rawSmemPtr = getStridedElementPtr(loc, op.getSrcMemref().getType(),
                                          srcMemref, indices, rewriter);

  // 2. 关键地址空间转换: ldmatrix 硬件指令强制要求传入 32 位 Shared Memory 相对偏移指针 (addrspace 3)
  // 若传入通用 64 位地址 (addrspace 0)，ptxas 会拒绝编译或插入昂贵的 cvta 转换
  Type smemPtrTy = LLVM::LLVMPointerType::get(rewriter.getContext(), /*addrSpace=*/3);
  Value smemPtrAS3 = rewriter.create<LLVM::AddrSpaceCastOp>(loc, smemPtrTy, rawSmemPtr);

  // 3. 根据加载矩阵块数量 (numTiles = 1, 2, 4) 与是否硬件转置 (.trans) 选择对应的 NVVM Intrinsic
  // 例如: @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.trans.b16
  Intrinsic::ID intrinsicId = getLdMatrixIntrinsicID(op.getNumTiles(), op.getTranspose());
  rewriter.replaceOpWithNewOp<LLVM::CallIntrinsicOp>(
      op, op.getResult().getType(), intrinsicId, ValueRange{smemPtrAS3});
  return success();
}
```

* **逐行机械执行分析**：
  1. **为什么 `ldmatrix` 仅使用前 $8 / 16 / 32$ 个线程的地址？**：`ldmatrix.m8n8.x4` 一次性加载 4 个 $8 \times 8$ 的 FP16 矩阵（共 32 行，每行 8 个 FP16 = 16 字节 = 128-bit）。硬件设计规定：Warp 内的 Thread $0 \dots 31$ 各自提供一个 **128-bit 行首地址**，内存控制器在一拍内从 Shared Memory 并行抓回 32 个 128-bit 行向量（共 512 字节），并在内部交叉开关中自动将每个 $8 \times 8$ 矩阵按每线程 2 个 FP16（1 个 32 位寄存器）的布局分发给全 Warp 所有 32 个线程。如果是 `.x1`（仅 1 个 $8 \times 8$ 矩阵，共 8 行），硬件仅读取 Thread $0 \dots 7$ 的地址寄存器，Thread $8 \dots 31$ 的地址输入被硬件直接忽略（因此无需对非活跃线程做越界地址保护！）。
  2. **免费硬件转置（`.trans` 修饰符）**：对于矩阵乘法 $C = A \times B$ 中的 $B$ 矩阵（在全局显存中按行优先存储，但 Tensor Core 需要按列索引 $K$ 维读取），加上 `.trans` 修饰符后，交叉开关在分发数据进寄存器的瞬间自动完成 $8 \times 8$ 局部块内转置，彻底消除寄存器内手动洗牌开销。

---

### 5. 【AI 编译器长序列注意力与 Ring/Paged KV-Cache 访存优化】vLLM / FlashDecoding 中的 PagedAttention 间接寻址向量化与 Block Table 预取合并
* **题目描述**：在大语言模型长上下文推理服务（如 vLLM / SGLang）中，KV Cache 并非连续存储，而是按固定大小（如 `BLOCK_SIZE = 16`）切分为离散物理页，通过二维页表 `block_tables[batch_idx, logical_block_idx]` 进行间接寻址。在 Triton 或自定义 CUDA 内核中，传统实现每个 Token 都要先读取页表再计算物理显存地址，导致严重的指针追逐延迟（Pointer Chasing）和合并访存断裂。请结合 Triton / FlashInfer 编译器与算子优化架构分析：编译器是如何通过将物理页大小 `BLOCK_SIZE` 与硬件 128 字节缓存行对齐、利用标量寄存器预取页表项并广播（Scalar Broadcast）、以及沿 Head Dimension 实施宽向量化加载（`v4.f16` / `128-bit`），从而在离散页表寻址下依然实现接近理论峰值的显存带宽利用率的？
* **核心架构直达**：
  * vLLM `paged_attention_v1/v2` & FlashInfer / Triton PagedAttention 算子内核
* **PagedAttention 页表标量预取与宽向量合并访存核心伪代码**：

```cpp
// 典型的 PagedAttention 离散页表高性能访存与向量化调度模型
void emitPagedAttentionKVCacheLoad(OpBuilder &b, Location loc,
                                   Value kCacheBasePtr, Value blockTablesPtr,
                                   Value logicalBlockIdx, Value headIdx,
                                   int headDim, int blockSize /* e.g., 16 */) {
  // 1. 关键硬件对齐原则: 将物理页大小 BLOCK_SIZE 设为 16 (对于 FP16 且 HeadDim=128，单页单头恰好占 16 * 128 * 2B = 4KB)
  // 且同一个 Warp (32 线程) 协同处理同一个 KV Head 的连续 Token 切片，确保全 Warp 命中相同的 logicalBlockIdx！

  // 2. 标量/Uniform 读取物理页号 (Physical Block Number):
  // 由于同 Warp 线程的 logicalBlockIdx 完全一致，编译器通过 UniformityAnalysis 将页表读取提升至标量总线！
  // 仅发射 1 次标量全局加载 (LDG.E.U32 / S_LOAD_DWORD) 并零延迟广播至全 Warp 32 线程:
  Value physicalBlockId = emitUniformScalarLoad(b, loc, blockTablesPtr, logicalBlockIdx);

  // 3. 计算物理页基地址 (Base Address of Physical KV Page):
  // pageBasePtr = kCacheBasePtr + physicalBlockId * (numKvHeads * blockSize * headDim) + headIdx * (blockSize * headDim)
  Value pageBasePtr = computePhysicalPageBase(b, loc, kCacheBasePtr, physicalBlockId, headIdx);

  // 4. 在物理页内部执行 100% 连续合并的 128-bit 宽向量加载！
  // 例如: 32 个线程横向覆盖 HeadDim=128 (每个线程负责连续 4 个 FP16 = 8 字节，或 16 线程组负责 128-bit 宽指令)
  // 彻底消除“每个元素查一次页表”的指针追逐灾难！
  Value kVecFrag = emitCoalescedVectorLoad128(b, loc, pageBasePtr, threadLocalOffset);
}
```

* **逐行机械执行分析**：
  1. **消除指针追逐序列化（Breaking Pointer Chasing）**：如果按 Token 维度将不同逻辑 Token 分配给同一 Warp 的不同线程，32 个线程会同时查询 32 个不同的 `physicalBlockId`，导致内存控制器发出 32 个互不连续的 DRAM 事务，有效带宽跌破 10%。编译器通过**线程映射重排（Thread-to-HeadDim Mapping）**，让整个 Warp 集中火力横向并行读取**同一个物理页内的同一个 Token 向量（如 `HeadDim = 128`）**。
  2. **Uniform 标量页表预取**：由于全 Warp 访问同一个物理页，页表读取 `block_tables[batch_idx, logical_block_idx]` 退化为全 Warp 一致的标量操作（Uniform Load），数据直接进入只读常量/标量缓存并广播给所有线程，页表查询延迟被完全隐藏在软件流水线的前置阶段。
  3. **页内满带宽突发传输（Intra-Page Burst Coalescing）**：在锁定 `pageBasePtr` 后，页内连续 16 个 Token 的存储布局在物理显存中是严格连续的，LSU 得以全力发射 `LDG.E.128`（128-bit 宽向量加载）与 `cp.async.cg`，使离散分页存储的 PagedAttention 达到了与连续显存 FlashAttention 几乎相同的理论带宽利用率。
