# CUDA 显存去重 gKSM 项目说明

## 背景与目标
- 通过拦截 CUDA Driver API（自定义 `libcuda.so`）在页粒度进行跨进程显存去重，降低多实例/多用户推理的显存占用。
- 利用 GPU 侧快速哈希、布隆过滤器、用户态缺页处理与进程间共享，尽量降低去重开销。
- 当前主要面向 A100、CUDA 12.4、570 驱动环境（参考仓库顶层 `README` 记录）。

## 目录速览
- `libcudahook.cpp/.h`：核心 Hook 逻辑，拦截 `cuMemAlloc/cuMemcpy/cuMemCreate/cuMemMap` 等；负责页跟踪、稳定性检测、去重与统计。
- `tools.cu/.h`：GPU/CPU 哈希、布隆过滤器、计时与统计；含批量 GPU 哈希内核与内存池复用。
- `ipc.cpp/.h`：UNIX 域套接字 + 共享内存元数据，用于跨进程共享/转移页所有权、发送去重请求。
- `hook.conf`：运行时配置（是否启用去重、扫描间隔、页大小、共享内存名、原/Hook 库路径等）。
- 构建与运行脚本：`make.sh`（编译并生成 `build/libcuda.so.1`）、`init.sh`（环境准备）、`run.sh`（示例基准），`vllm/` 下有推理脚本示例。
- 其他：`cuda_hook.c`、`ipc.cpp` 辅助实现，`tools.h` 中包含统计枚举与全局配置结构体。

## 工作原理概述
- **库替换与 API 拦截**：通过 `patchelf` 包装原生 `libcuda.so.1`，加载时执行构造函数初始化 Hook、读取配置、建立共享内存与 IPC。
```
37:68:libcudahook.cpp
__attribute__((constructor))
static void init_hook() {
    ...
    parse_config_file(config_file_path, g_hook_config);
    g_page_map = new std::map<CUdeviceptr, LocalPageInfo>();
    init_hook_api();
    init_ipc();
}
```
- **页跟踪与稳定性检测**：在 `cuMemAlloc/cuMemMap` 等分配时登记页；后台线程按批扫描未去重页，计算哈希、判断“稳定页”后尝试去重。
- **GPU 加速哈希**：在 `tools.cu` 中实现单页与批量 XXH64 GPU kernel，并复用统一的 GPU 内存池与流，降低额外开销。
- **布隆过滤 + 候选列表**：对快速哈希结果做预过滤，减少重复计算；候选页列表用于跨进程、跨分配批次匹配。
- **跨进程共享/去重**：通过 `cuMemExportToShareableHandle` + UNIX FD 传递，将去重页映射到其他进程；支持所有权转移与去重请求回调。
- **用户态缺页处理（可选）**：配置 `userfault_fd_path` 时，用户态处理缺页并按需映射/复制。

## 构建与安装
1. 先备份并设置原生 CUDA 库（脚本会在 `/usr/lib/x86_64-linux-gnu` 下复制为 `libcuda-native.so`）。
2. 执行构建（需要 `nvcc`、OpenSSL、xxhash、CUDA 12.4）：
   ```bash
   cd /root/dedup
   bash make.sh
   ```
3. 确认生成 `build/libcuda.so.1`，并确保 `LD_LIBRARY_PATH=/root/dedup/build` 在目标进程可见。
4. 按需更新 `hook.conf`，删除旧的共享内存文件：`rm /dev/shm/cuda_dedup_shared`。

## 关键能力与特性
- 跨进程显存去重：利用 CUDA VMM 共享句柄与 FD 传递，实现页级去重与所有权转移。
- GPU 加速哈希：批量 XXH64/自定义超快哈希，减少 CPU 参与和 PCIe 往返。
- 稳定页判定：多轮哈希一致后才去重，避免频繁写导致的抖动。
- 布隆过滤：预筛重复页，降低哈希/对比成本。
- 统计与可观测性：Hook API 调用次数、耗时汇总，便于评估收益与瓶颈。


