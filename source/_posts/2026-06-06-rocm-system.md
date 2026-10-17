---
layout: single
title: "Compile rocm runtime library"
date: 2026-06-06
tags: [ROCm]
---

## 引言

AMD开源了runtime library, 如果你有一张AMD的卡， 完全可以自己尝试编译一下他们的仓库，debug一下


## 步骤

克隆仓库: `https://github.com/ROCm/rocm-systems.git`和`https://github.com/ROCm/llvm-project.git`, 我自己编译的版本是rocm-7.2, 注意分支的一致性

### 编译LLVM

下面的路径可以按需替换

```sh

cmake -S llvm -B build -DCMAKE_BUILD_TYPE=Release -G Ninja \
      -DLLVM_ENABLE_PROJECTS="clang;lld" \
      -DLLVM_EXTERNAL_PROJECTS="device-libs;comgr" \
      -DLLVM_EXTERNAL_DEVICE_LIBS_SOURCE_DIR=/home/jing.zhou/rocm-llvm/amd/device-libs \
      -DLLVM_EXTERNAL_COMGR_SOURCE_DIR=/home/jing.zhou/rocm-llvm/amd/comgr \
      -DCMAKE_INSTALL_PREFIX=/home/jing.zhou/rocm-systems/projects/install \
      -DLLVM_TARGETS_TO_BUILD="X86;AMDGPU" \
      -DCMAKE_LINKER=/home/jing.zhou/tools/bin/mold

```

### 编译clr

cd到`rocm-systems`目录，

```sh
export CLR_DIR="$(readlink -f projects/clr)"
export HIP_DIR="$(readlink -f projects/hip)"

export PATH=/home/jing.zhou/rocm-systems/projects/install/bin:$PATH

cd "$CLR_DIR"
mkdir -p build; cd build
cmake -DHIP_COMMON_DIR=$HIP_DIR -DCMAKE_PREFIX_PATH="/home/jing.zhou/rocm-systems/projects" -DROCM_INSTALL_PATH="/home/jing.zhou/rocm-systems/projects/install" -DCMAKE_INSTALL_PREFIX=/home/jing.zhou/rocm-systems/projects/install -DCLR_BUILD_HIP=ON -DCLR_BUILD_OCL=OFF -DHIP_PLATFORM=amd ..
make -j$(nproc)
make install

```

### demo

```sh
export LD_LIBRARY_PATH=/home/jing.zhou/rocm-systems/projects/install/lib:$LD_LIBRARY_PATH
export HIP_CLANG_PATH="/home/jing.zhou/rocm-systems/projects/install/bin"
export HIP_PATH="/home/jing.zhou/rocm-systems/projects/install"
export HIP_PLATFORM=amd
```

写一份简单的hello-world代码

```cpp
#include <iostream>
// 引入 HIP 运行时头文件
#include <hip/hip_runtime.h>

// 1. 定义 GPU 核函数 (Kernel)
// __global__ 告诉编译器这个函数是在 GPU 上执行的，但由 CPU 发起调用
__global__ void gpu_hello_world() {
  // blockIdx, blockDim, threadIdx 是 HIP 的内置变量
  // 这里计算出当前线程的全局唯一 ID
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;

  // printf 在 GPU 核函数中是受支持的（适合调试）
  printf("Hello World from GPU thread %d!\n", gtid);
}

int main() {
  // 2. 在 CPU (Host) 上打印
  std::cout << "Hello World from CPU!" << std::endl;

  // 3. 配置 GPU 的执行网格 (Grid) 和线程块 (Block)
  // 这里我们启动 1 个线程块，里面包含 4 个线程
  dim3 grid(1);
  dim3 block(4);

  // 4. 启动 GPU 核函数
  // 使用 <<<...>>> 语法来指定网格和线程块大小
  hipLaunchKernelGGL(gpu_hello_world, grid, block, 0, 0);

  // 5. 同步 CPU 和 GPU
  // 因为 GPU 的执行是异步的，CPU 需要等待 GPU 打印完毕后再退出程序
  hipDeviceSynchronize();

  return 0;
}

```
输出

```
Hello World from CPU!
Hello World from GPU thread 0!
Hello World from GPU thread 1!
Hello World from GPU thread 2!
Hello World from GPU thread 3!

```
