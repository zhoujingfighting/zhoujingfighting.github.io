# **GPGPU 编译器模拟面试与深度解析 —— Day 05**

# **1\. MachineLICM 在 GPU 上的“发散不变式”陷阱与源码剖析**

## **题目描述**

在 GPGPU 编译器的机器级循环不变式外提（MachineLICM）优化 pass 中，如果一个在循环体内表现为“循环不变”的表达式实际上是发散的（Divergent，例如直接或间接依赖于 `threadIdx`、`lane_id` 或 warp 内部的分支执行掩码），盲目将其外提至 Preheader 块会引发什么致命问题？GPU 后端编译器是如何结合 Uniformity Analysis 在 Code Motion 阶段进行精准过滤的？

## **源码级参考解析**

1. **源码文件直达**：  
   * `llvm/lib/CodeGen/MachineLICM.cpp`  
   * `llvm/include/llvm/CodeGen/MachineRegisterInfo.h`  
   * `llvm/include/llvm/Analysis/UniformityAnalysis.h`  
2. **核心类与接口剖析**：  
   * `MachineLICMBase`：循环不变式外提的核心驱动类，继承自 `MachineFunctionPass`。  
   * `MachineLICMBase::IsLoopInvariant(MachineInstr &MI)`：遍历 `MI.operands()`，通过 `MRI->getVRegDef(Reg)` 验证其所有寄存器操作数是否由当前循环（`CurLoop`）之外定义。  
   * `MachineLICMBase::IsProfitableToHoist(MachineInstr &MI)`：调用 `RegPressureTracker` 计算外提该指令引入的跨边界虚拟寄存器生命周期对寄存器水位的压力影响（`PressureDiff`）。  
3. **逐步源码逻辑拆解**：  
   * **不变式提取流程**：在 `ProcessLoop` 遍历基本块时，`IsLoopInvariant` 递归检查 Def-Use 链。若一条指令的所有源操作数均为不变式，该指令被加入可提升候选集。  
   * **发散性冲突判定（Divergence Legality）**：在 GPU 目标（如 AMDGPU/NVPTX）中，若指令依赖发散源，虽然其操作数对单一 Lane 在迭代间不变，但 Lane 之间的值各异。循环体内的该指令受局部活跃掩码（`EXEC` / Predicate）约束；若外提至支配循环的 Preheader，Preheader 运行在父级上下文的全掩码或不同掩码下。这会导致非活跃 Lane 错误执行副作用或覆盖寄存器。因此 GPU 后端在 `isProfitableToHoist` 或重写的准入检查中调用 `UniformityInfo::isUniform(&MI)`：若为发散指令且该指令所在基本块不能后支配（Post-Dominate）Preheader，则直接阻断外提。  
   * **寄存器压力截断**：外提指令会将其目标寄存器的存活区间拉长至覆盖整个循环。`IsProfitableToHoist` 调用 `TII->getRegClass(MRI->getRegClass(Reg))`，若增加的寄存器导致当前 SM 的寄存器占用量超过阈值（引发 Occupancy 下滑），即便是不变式也强制放弃外提。

---

# **2\. I-Cache Thrashing 与激进内联的博弈与源码剖析**

## **题目描述**

在 CPU 上，函数内联通常能通过消除调用开销和扩大过程间优化空间带来显著提升。然而在 GPU 上，过度激进的内联会导致指令缓存抖动（I-Cache Thrashing）。请详细解释其硬件层面的传导机制，并说明 GPU 编译器在设计 Inliner Cost Model 时，如何精准量化 Code Size 膨胀对 I-Cache 和 TLP 的负面效应？

## **源码级参考解析**

1. **源码文件直达**：  
   * `llvm/lib/Analysis/InlineCost.cpp`  
   * `llvm/include/llvm/Analysis/InlineCost.h`  
   * `llvm/lib/Target/NVPTX/NVPTXTargetTransformInfo.cpp`  
   * `llvm/lib/Target/AMDGPU/AMDGPUTargetTransformInfo.cpp`  
2. **核心类与接口剖析**：  
   * `InlineCostCallAnalyzer`：负责遍历 Callee 中的基本块和指令，计算累加代价。  
   * `TargetTransformInfo::getInlineCostThreshold()`：目标后端定制内联阈值的钩子函数。  
3. **逐步源码逻辑拆解**：  
   * **Cost 累加机制**：`InlineCostCallAnalyzer::analyze()` 遍历 Callee 的 Instruction 流，通过 `TTI.getInstructionCost(&I)` 为每条指令累加 Code Size 成本。  
   * **GPU 专属惩罚注入**：  
     * 在 `AMDGPUTTIImpl::getInlineCostThreshold()` 和 `NVPTXTTIImpl` 中，GPU 后端将默认的基准阈值大幅调低，严格限制大函数内联。  
     * 若 Callee 内包含 `AllocaInst`（局部栈内存申请）或存在深层循环，`InlineCostCallAnalyzer` 会施加巨大的惩罚倍数，因为局部变量在 GPU 上会映射到高延迟的 Local Memory。  
   * **Occupancy 感知拦截**：GPU 后端结合函数属性中的 `amdgpu-flat-work-group-size` 或 `__launch_bounds__` 预估目标寄存器用量。若内联会导致 Callee 中的大量局部变量生命周期叠加入 Caller，导致单线程寄存器用量跨越阶梯临界点，分析器直接返回 `InlineCost::getNever("exceeds register budget")` 终止内联。

---

# **3\. Live Range Splitting 与 GPU 专属 Spill Cost 评估源码剖析**

## **题目描述**

通用的 CPU 寄存器分配算法在处理变量生命周期切分（Live Range Splitting）和 Spill 时，通常假设 Spill 成本为固定的 L1 Cache 或 RAM 访问延迟。为什么这种假设在 GPU 编译器中会导致性能灾难？GPU 后端在 Live Range Splitting 和 Spill Cost Model 中引入了哪些专属评估指标？

## **源码级参考解析**

1. **源码文件直达**：  
   * `llvm/lib/CodeGen/RegAllocGreedy.cpp`  
   * `llvm/lib/CodeGen/LiveIntervals.cpp`  
   * `llvm/lib/CodeGen/CalcSpillWeights.cpp`  
2. **核心类与接口剖析**：  
   * `VirtRegAuxInfo::weightCalcHelper()`：计算每个虚拟寄存器 Spill Weight 的核心实现。  
   * `RAGreedy::selectOrSplit(LiveInterval &VirtReg, SmallVectorImpl<Register> &SplitVRegs)`：当物理寄存器着色失败时，决定是 Split 还是 Spill 的主循环。  
3. **逐步源码逻辑拆解**：  
   * **权重计算公式**：在 `CalcSpillWeights.cpp` 中，基础公式为：  
     * \$\$\\text{Weight} \= \\frac{\\sum (\\text{Use/Def 频次} \\times \\text{LoopDepthWeight})}{\\text{LiveRange 长度}}\$\$  
     * 其中 `LoopDepthWeight` 按 \$10^{\\text{Depth}}\$ 指数放大循环内部访存的代价。  
   * **GPU 内存放大因子**：  
     * CPU 上单次溢出仅写入一次 L1 Cache（4 字节）。GPU 上一个 Spill 由 Warp 内 32 个线程并发触发（\$32 \\times 4 \= 128\$ 字节）。若指针非连续，硬件将其拆散为数十次独立的 DRAM 事务。  
     * 后端通过 `TargetRegisterInfo::getSpillSize()` 量化向量及标量溢出惩罚，优先保留跨越深层循环的寄存器。  
   * **切分与重计算协同（Split vs Remat）**：  
     * 在 `RAGreedy::splitAroundRegion()` 中，若变量在循环体外定义、循环体内高频使用，分配器尝试将活跃区间切分为“循环外长区间”与“循环内短区间”，避免将全函数生命周期绑定在单一物理寄存器上。  
     * 若 `TII->isRematerializable(*MI)` 成立，`InlineSpiller::reMaterializeFor()` 直接就地重算指令，将 Spill 权重抹平，彻底免除显存读写。

---

# **4\. Tensor Lifetime Scheduling 与显存规划源码剖析**

## **题目描述**

在 AI 编译器的图优化层，Tensor Lifetime Scheduling 是将模型内存占用降至最低的核心技术。请简述基于生命周期分析与区间图着色进行显存复用的基本算法逻辑。面对动态控制流时，编译器如何做出高效规划？

## **源码级参考解析**

1. **源码文件直达**：  
   * `mlir/lib/Dialect/Bufferization/Transforms/BufferDeallocationPass.cpp`  
   * `mlir/lib/Dialect/Bufferization/IR/BufferDeallocationOpInterface.cpp`  
2. **核心类与接口剖析**：  
   * `BufferDeallocation`：负责在 MLIR 降级阶段分析 `memref` 的生命周期边界并自动插入 `memref.dealloc`。  
   * `BufferDeallocationState`：基于控制流图（CFG）维护所有已分配内存块的所有权（Ownership）与逃逸分析（Escape Analysis）。  
3. **逐步源码逻辑拆解**：  
   * **区间构建（Interval Collection）**：对计算图拓扑排序，为每个中间 Tensor 标定 Birth（生成算子序号）与 Death（最后一个消费者读取完毕序号）。  
   * **活跃区间冲突图着色**：将无重叠区间的 Tensor 分配到同一物理显存 Offset。在 Greedy First-Fit 算法中，维护一个空闲内存块堆（Memory Arena），按时间切片检索未被占用的显存槽。  
   * **分支与循环的作用域归纳**：  
     * 在 `BufferDeallocation::handleBlock` 中，若遇到条件分支（`cf.br` / `scf.if`），编译器采用保守的作用域提升（Scope Promotion）：若 Tensor 在分支内创建但在分支汇聚点（Join Block）仍存活，则将所有权移交至共同的支配前驱块。  
     * 循环内部生成的中间 Tensor，若其生存期局限在单次迭代内部，在每次迭代末尾显式插入释放指令，实现跨迭代的显存重用。

# **5\. GPU 异常（Trap/Assert）与编译器底层支撑源码剖析**

## **题目描述**

在没有完整 OS 中断处理机制的 GPU 上，当执行触发断言失败或越界访问时，编译器必须在指令生成、寄存器保留和运行时协议上做出哪些特殊设计，以保证安全报错且不导致硬件挂起？

## **源码级参考解析**

1. **源码文件直达**：  
   * `llvm/lib/Target/NVPTX/NVPTXISelLowering.cpp`  
   * `llvm/lib/Target/AMDGPU/SIISelLowering.cpp`  
   * `llvm/lib/CodeGen/SelectionDAG/SelectionDAGBuilder.cpp`  
2. **核心类与接口剖析**：  
   * `SelectionDAGBuilder::visitTrap()`：将通用 LLVM IR 原语 `@llvm.trap` 转换为目标架构的硬件异常指令。  
   * `NVPTXTargetLowering::LowerOperation(SDValue Op, SelectionDAG &DAG)`：处理 `ISD::TRAP` 的目标特化降级。  
3. **逐步源码逻辑拆解**：  
   * **IR 降级与指令生成**：  
     * NVPTX 后端将 `ISD::TRAP` 节点直接发射为 PTX 汇编指令 `trap;`；在断言（Assert）场景下，降级生成带条件的分支和硬件断点 `brkpt;`。  
     * AMDGPU 后端将其降级为 `s_trap <trap_code>`，该指令触发硬件 Trap Handler。  
   * **控制流截断（Unreachable Termination）**：  
     * 编译器在发射 `TRAP` 后紧跟 `ISD::TargetIndex` 生成终止标记，并调用 `visitUnreachable()` 告知调度器截断当前的 CFG 分支，禁止后续任何普通指令与其重排，防止脏写污染。  
   * **参数与元数据传递**：  
     * 在高级框架（如 CUDA SDK / ROCm）中，`assert()` 宏扩展为先调用内部格式化函数将文件名与行号存入系统预留的固定地址缓冲区（System Ring Buffer），随后执行原子置位，最后发射硬件 `trap` 指令挂起 Warp。Host 端 Driver 在 `cudaDeviceSynchronize()` 返回失败时，回读此全局缓冲区解析出精准的崩溃位置。