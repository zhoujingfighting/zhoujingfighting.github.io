---
layout: single
title: "Hello World from HIP and LLVM"
date: 2026-06-06
categories: [Compiler, HIP]
tags: [GPU, ROCm]
---

### 引言

这是一篇用来测试博客是否渲染成功的硬核技术文章。

### 核心代码测试

```cpp
#include <hip/hip_runtime.h>
__global__ void hello() {
    printf("Hello from GPU!\n");
}
```
