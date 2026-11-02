# **GPGPU 编译器模拟面试与深度解析 —— Day 08**

# **1\. MLIR Dialect Lowering 与 GPU 目标属性保留**

## **题目描述**

在降级到 `llvm` Dialect 时，MLIR 如何保留地址空间、Kernel 调用约定及线程内置函数？

## **源码级深度剖析**

* **源码文件直达**：  
  * `mlir/lib/Conversion/GPUToLLVM/GPUToLLVMPass.cpp`  
  * `mlir/lib/Conversion/GPUCommon/GPUOpsLowering.cpp`  
* **核心类与核心代码块逻辑**：

```c
// mlir/lib/Conversion/GPUCommon/GPUOpsLowering.cpp
LogicalResult GPUFuncOpLowering::matchAndRewrite(
    gpu::GPUFuncOp gpuFuncOp, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // 1. 检查是否带有 gpu.kernel 属性
  if (!gpuFuncOp.isKernel())
    return failure();

  // 2. 映射为对应 Target 的调用约定与属性
  auto llvmFuncType = typeConverter->convertType(gpuFuncOp.getFunctionType());
  auto llvmFuncOp = rewriter.create<LLVM::LLVMFuncOp>(
      gpuFuncOp.getLoc(), gpuFuncOp.getName(), llvmFuncType);

  // 3. 设置 NVVM / ROCDL 特化属性
  if (isNVVM) {
    llvmFuncOp->setAttr("nvvm.kernel", rewriter.getUnitAttr());
    llvmFuncOp->setAttr("nvvm.maxntid", gpuFuncOp->getAttr("known_block_size"));
  }
  rewriter.replaceOp(gpuFuncOp, llvmFuncOp);
  return success();
}
```

* **逐行逻辑深度拆解**：  
  * **调用约定绑定**：将 `gpu.kernel` 单元属性直接翻译为 `nvvm.kernel` 或 AMD 的 `amdgpu_kernel` (CallingConv 85)。  
  * **微架构硬件限制透传**：将 Block 尺寸映射为目标后端元数据，防止机器指令生成时误设过高寄存器限制。

---

# **2\. Tensor 到 Memref 的 Bufferization 与 GPU 共享内存映射**

## **题目描述**

In-place Bufferization 如何复用内存？Shared Memory 是如何分配并转换为底层 LLVM IR 的？

## **源码级深度剖析**

* **源码文件直达**：  
  1. `mlir/lib/Dialect/Bufferization/Transforms/OneShotAnalysis.cpp`  
  2. `mlir/lib/Conversion/GPUCommon/GPUOpsLowering.cpp`  
* **核心类与核心代码块逻辑**：

```c
// MLIR Shared Memory Buffer Allocation 核心映射逻辑
Value allocateSharedMemory(OpBuilder &builder, Location loc, MemRefType type) {
  // 静态 Shared Memory 映射为 AddressSpace 3 的内部全局变量
  auto globalOp = builder.create<LLVM::GlobalOp>(
      loc, type, /*isConstant=*/false, LLVM::Linkage::Internal,
      "smem_alloc", Attribute(), /*alignment=*/16, /*addrSpace=*/3);
  return builder.create<LLVM::AddressOfOp>(loc, globalOp);
}
```

* **逐行逻辑深度拆解**：  
  1. **就地复用**：生命周期分析确定无冲突后就地覆盖，消除显存分配。  
  2. **LDS 映射**：`gpu.dynamic_shared_memory` 映射为外部动态共享内存声明指针。

---

# **3\. SCF / Affine 循环到 GPU Grid/Block 的映射**

## **题目描述**

嵌套循环如何映射到 GPU 的二维/三维线程拓扑？对下游循环展开有何影响？

## **源码级深度剖析**

* **源码文件直达**：  
  1. `mlir/lib/Conversion/AffineToGPU/AffineToGPU.cpp`  
  2. `mlir/lib/Dialect/GPU/Transforms/ParallelLoopToGpu.cpp`  
* **核心类与核心代码块逻辑**：

```c
// 循环归纳变量重写为 GPU 线程索引逻辑
Value rewriteInductionVariable(OpBuilder &b, Location loc, Value lowerBound, 
                               Value step, Value blockId, Value threadId, Value blockDim) {
  Value threadOffset = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, blockId, blockDim), threadId);
  return b.create<arith::AddIOp>(
      loc, lowerBound, b.create<arith::MulIOp>(loc, threadOffset, step));
}
```

* **逐行逻辑深度拆解**：  
  1. **映射机制**：通过 Pass 将循环归纳变量重写为全局线程索引计算公式并嵌入 `gpu.launch`。  
  2. **影响**：破坏原本连续循环边界，但为后端向量化和 SIMT 谓词化建立硬件基础。

---

# **4\. MLIR Pattern Rewriting 与硬件原语对接**

## **题目描述**

解释 Pattern Rewriting 的匹配收敛机制。当矩阵 Shape 不支持时如何 Fallback？

## **源码级深度剖析**

* **源码文件直达**：  
  * `mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp`  
* **核心类与核心代码块逻辑**：

```c
// mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp
LogicalResult GreedyPatternRewriteDriver::processWorklist() {
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    for (const auto &pattern : patterns) {
      if (pattern->matchAndRewrite(op, rewriter).succeeded()) {
        // 重写成功，重新将关联节点入队以驱动收敛
        addToWorklist(op->getUsers());
        break;
      }
    }
  }
  return success();
}
```

* **逐行逻辑深度拆解**：  
  * **匹配驱动**：`GreedyPatternRewriteDriver` 维护 Worklist 反复应用规则直到达到固定点。  
  * **Fallback**：无法匹配专有硬件指令（如 wmma）时，回退生成标量/向量循环和 FMA 累加。

---

# **5\. 混合编译架构下 MLIR 与传统 LLVM 后端的职责分工**

## **题目描述**

为什么底层优化不适合在 MLIR 高层完成？两者的理想交接点在哪里？

## **源码级深度剖析**

1. **核心分工逻辑图解**：

```
[ MLIR: 高层几何变换 (Tile/Fuse/Bufferize) ] 
       │ 
       ▼ (转换为纯净 llvm Dialect)
[ LLVM Backend: 物理寄存器分配 / SchedDAG / Hazard 防范 / 汇编发射 ]
```

2. **逐行逻辑深度拆解**：  
   * **底层不可替代**：寄存器分配和指令调度高度依赖微架构物理细节，高层无法精确评估压力。  
   * **交接点**：当高层结构化循环与内存分配完全降级为纯净的 llvm Dialect 时，交由 LLVM 后端做机器码生成。