# **GPGPU 编译器模拟面试与深度解析 —— Day 07**

# **1\. Kernel 函数参数传递与 Parameter Space 映射**

## **题目描述**

Host 端发起 Kernel 调用时，参数是如何物理传递至 GPU 硬件端的？在 LLVM 后端 lowering 阶段，TargetLowering 接口与 Calling Convention（如 NVPTX 的 PTX\_Kernel）如何将抽象的函数参数映射到具体的 Parameter Address Space（`addrspace(4)` 或 Constant/Param 寄存器）？大型结构体与对齐约束如何降级？

## **源码级深度剖析**

* **源码文件直达**：  
  * `llvm/lib/Target/NVPTX/NVPTXISelLowering.cpp`  
  * `llvm/lib/CodeGen/SelectionDAG/SelectionDAGBuilder.cpp`  
* **核心类与核心代码块逻辑**：  
* 

```c
// llvm/lib/Target/NVPTX/NVPTXISelLowering.cpp
SDValue NVPTXTargetLowering::LowerFormalArguments(
    SDValue Chain, CallingConv::ID CallConv, bool isVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &dl,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals) const {
  // 遍历所有 Kernel 参数
  for (unsigned i = 0, e = Ins.size(); i != e; ++i) {
    Type *ArgTy = Ins[i].ArgVT.getTypeForEVT(*DAG.getContext());
    // 强制转换为 Parameter 专用地址空间 (NVPTX::PTX_PARAM_SPACE)
    SDValue ParamVal = DAG.getLoad(
        Ins[i].ArgVT, dl, Chain,
        DAG.getTargetConstant(Ins[i].PartOffset, dl, MVT::i32),
        MachinePointerInfo(AddressSpace::Param));
    InVals.push_back(ParamVal);
  }
  return Chain;
}
```

* **逐行逻辑深度拆解**：  
  * **参数空间抽象**：在 `LowerFormalArguments` 中，所有入口参数被截获并映射到 `AddressSpace::Param`（PTX `.param`）。  
  * **复合结构体降级**：结构体被展开为偏移序列，若满足 16 字节对齐，生成向量加载指令（如 `ld.param.v4.b32`）一次性读取。

---

# **2\. Issue Port Starvation 与异构管线调度**

## **题目描述**

在现代 GPGPU 架构中，SM 内部包含多种异构计算管线。当某一类指令过载导致 Issue Port Starvation 时，编译器如何借助 TargetScheduleModel 与 Hazards Recognition 实现异构管线交织调度？

## **源码级深度剖析**

* **源码文件直达**：  
  * `llvm/lib/CodeGen/MachineScheduler.cpp`  
  * `llvm/include/llvm/CodeGen/TargetSchedule.h`  
* **核心类与核心代码块逻辑**：  
* 

```c
// llvm/lib/CodeGen/MachineScheduler.cpp
void ScheduleDAGMILive::initQueues(ArrayRef<SUnit *> TopRoots,
                                   ArrayRef<SUnit *> BotRoots) {
  // 追踪 ProcResource 压力
  for (unsigned i = 0; i < SchedModel.getNumProcResourceKinds(); ++i) {
    if (CurrentPressure[i] > CriticalResourceThreshold[i]) {
      // 当前端口（如 SFU 端口）发生饥饿/过载，提升其他异构管线指令优先级
      SchedBoundary::bumpNodeTop(SUnit *SU);
    }
  }
}
```

* **逐行逻辑深度拆解**：  
  * **端口压力追踪**：`SchedBoundary` 追踪每条发射指令对 `ProcResource` 的消耗百分比。  
  * **指令交织优化**：在密集 Memory 或 SFU 指令中穿插独立的 ALU/FMA 计算指令，打散同质指令丛，避免发射端口阻塞。

---

# **3\. GPGPU 上的间接函数调用与动态并行编译挑战**

## **题目描述**

* 在 GPU 上支持虚函数、函数指针及动态并行（CDP）面临严重的硬件与编译器挑战。编译器如何通过去虚拟化（Devirtualization）、跳转表及动态栈分配予以解决？

## **源码级深度剖析**

* **源码文件直达**：  
  * `llvm/lib/Transforms/IPO/Devirt.cpp`  
  * `llvm/lib/Target/NVPTX/NVPTXISelLowering.cpp`  
* **核心类与核心代码块逻辑**：  
* 

```c
// llvm/lib/Transforms/IPO/Devirt.cpp
bool Devirtualization::tryDevirtualize(CallBase &CB) {
  // 查找可能的 Callee 闭包候选集
  SmallVector<Function *, 4> Targets;
  if (findTargets(CB, Targets) && Targets.size() == 1) {
    // 替换间接调用为直接调用，消除 GPU 栈溢出与 Warp 分歧
    CB.setCalledFunction(Targets[0]);
    return true;
  }
  return false;
}
```

* **逐行逻辑深度拆解**：  
  * **消除分支发散**：间接调用可能导致同 Warp 线程跳转到不同代码段。通过 LTO 去虚拟化，将无法推导的函数指针重构为条件分支并在全 Warp 层面聚拢。  
  * **栈空间预分配**：CUDA Runtime 在编译期估算 `MAX_STACK_SIZE` 并注入 Kernel 元数据，驱动层提前为每个线程配置独立的 Local Memory 栈帧。

---

# **4\. LLVM IR atomicrmw 到弱一致性 GPU ISA 的转换**

## **题目描述**

GPU 架构采用弱内存模型。编译器如何根据 SyncScope 映射硬件原子指令修饰符？若漏配 Scope 或 Fence 会导致什么严重后果？

## **源码级深度剖析**

1. **源码文件直达**：  
   1. `llvm/lib/Target/NVPTX/NVPTXISelLowering.cpp`

   2. `llvm/lib/Target/AMDGPU/SIISelLowering.cpp`  
2. **核心类与核心代码块逻辑**：

```c
// llvm/lib/Target/NVPTX/NVPTXISelLowering.cpp
SDValue NVPTXTargetLowering::LowerATOMIC_RMW(SDValue Op, SelectionDAG &DAG) const {
  auto *AtomicNode = cast<AtomicSDNode>(Op);
  SyncScope::ID SSID = AtomicNode->getSyncScopeID();
  
  // 映射作用域修饰符 (.cta / .gpu / .sys)
  NVPTX::Scope ScopeQualifier = NVPTX::Scope::CTA;
  if (SSID == DAG.getContext()->getOrInsertSyncScopeID("agent"))
    ScopeQualifier = NVPTX::Scope::GPU;
  else if (SSID == DAG.getContext()->getOrInsertSyncScopeID("cross-system"))
    ScopeQualifier = NVPTX::Scope::SYS;

  return DAG.getNode(NVPTXISD::ATOM_ADD, SDLoc(Op), AtomicNode->getVTList(),
                     AtomicNode->getChain(), AtomicNode->getBasePtr(),
                     AtomicNode->getVal(), DAG.getTargetConstant(ScopeQualifier, SDLoc(Op), MVT::i32));
}
```

3. **逐行逻辑深度拆解**：  
   1. **硬件修饰符绑定**：`agent` 映射为 `.gpu`，指示 LSU 绕过本地 SM 的 L1 脏写，下发至全局 L2 协调点。  
   2. **防范数据损坏**：若误降级为 `.cta`，其他 SM 将持续读取过期的本地缓存行，引发静默数据竞争。

---

# **5\. Layout Propagation 与 Strided Memref 性能损耗**

## **题目描述**

非连续步长访存如何破坏合并访存（Coalesced Memory Access）？MLIR 中的 Layout Propagation 如何消除转置？遇到无法消除的步长访存时如何通过 Shared Memory 暂存转置恢复满带宽？

## **源码级深度剖析**

* **源码文件直达**：  
  * `mlir/lib/Dialect/MemRef/Transforms/ComposeSubView.cpp`

  * `mlir/lib/Dialect/Affine/Transforms/LoopFusion.cpp`  
* **核心类与核心代码块逻辑**：

```c
// MLIR Shared Memory Staging Transpose 伪代码结构
void emitSmemStagedTranspose(OpBuilder &b, Location loc, Value gmemInput) {
  // 1. 分配带 Padding 的片上二维 Tile: [32][33]
  auto smemType = MemRefType::get({32, 33}, b.getF32Type(), {}, 3);
  Value smemTile = b.create<memref::AllocOp>(loc, smemType);

  // 2. Warp 协同沿行连续读取 (100% Coalesced Global Read)
  b.create<gpu::SubgroupMmaLoadMatrixOp>(loc, smemTile, gmemInput);

  // 3. 按列读取转换为计算所需布局 (无 Bank 冲突)
  b.create<gpu::BarrierOp>(loc);
}
```

* **逐行逻辑深度拆解**：  
  * **带宽放大抑制**：步长为 \$N\$ 的访问导致 32 线程分散在 32 个 Cache Line，总线利用率骤跌至 3.125%。  
  * **Shared Memory 暂存中转**：Warp 沿全局内存物理连续方向做合并读取写入 SRAM，利用二维数组行填充消除列向读取时的 Bank 冲突，恢复 100% 满带宽。

---

