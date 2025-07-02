#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

// 原始函数指针
static cudaError_t (*orig_cudaMalloc)(void **, size_t) = NULL;
static cudaError_t (*orig_cudaFree)(void *) = NULL;
static cudaError_t (*orig_cudaMemcpy)(void *, const void *, size_t, enum cudaMemcpyKind) = NULL;
static cudaError_t (*orig_cudaMemset)(void *, int, size_t) = NULL;
static cudaError_t (*orig_cudaLaunchKernel)(const void *, dim3, dim3, void **, size_t, cudaStream_t) = NULL;
static cudaError_t (*orig_cudaDeviceSynchronize)(void) = NULL;
static cudaError_t (*orig_cudaGetLastError)(void) = NULL;

// 资源统计
static size_t total_allocated = 0;
static size_t current_allocated = 0;
static size_t peak_allocated = 0;
static size_t memcpy_bytes = 0;
static unsigned int kernel_launches = 0;

// 分配追踪
typedef struct {
    void* ptr;
    size_t size;
} Allocation;
static Allocation* allocations = NULL;
static size_t alloc_count = 0;
static size_t alloc_capacity = 0;

// 获取当前时间戳
static const char* get_timestamp() {
    static char buffer[64];
    time_t now = time(NULL);
    struct tm *tm_info = localtime(&now);
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", tm_info);
    return buffer;
}

// 初始化原始函数指针
__attribute__((constructor))
static void init_orig_functions() {
    // 显式加载 CUDA 运行时库
    void *cudart = dlopen("/usr/local/cuda/lib64/libcudart.so", RTLD_NOW | RTLD_GLOBAL);
    if (!cudart) {
        fprintf(stderr, "[%s] ERROR: Failed to load libcudart.so: %s\n", 
                get_timestamp(), dlerror());
        return;
    }
    
    // 获取原始函数地址
    orig_cudaMalloc = dlsym(RTLD_NEXT, "cudaMalloc");
    orig_cudaFree = dlsym(RTLD_NEXT, "cudaFree");
    orig_cudaMemcpy = dlsym(RTLD_NEXT, "cudaMemcpy");
    orig_cudaMemset = dlsym(RTLD_NEXT, "cudaMemset");
    orig_cudaLaunchKernel = dlsym(RTLD_NEXT, "cudaLaunchKernel");
    orig_cudaDeviceSynchronize = dlsym(RTLD_NEXT, "cudaDeviceSynchronize");
    orig_cudaGetLastError = dlsym(RTLD_NEXT, "cudaGetLastError");
    
    // 检查是否成功获取所有函数
    if (!orig_cudaMalloc) fprintf(stderr, "Failed to hook cudaMalloc\n");
    if (!orig_cudaFree) fprintf(stderr, "Failed to hook cudaFree\n");
    if (!orig_cudaMemcpy) fprintf(stderr, "Failed to hook cudaMemcpy\n");
    if (!orig_cudaMemset) fprintf(stderr, "Failed to hook cudaMemset\n");
    if (!orig_cudaLaunchKernel) fprintf(stderr, "Failed to hook cudaLaunchKernel\n");
    if (!orig_cudaDeviceSynchronize) fprintf(stderr, "Failed to hook cudaDeviceSynchronize\n");
    if (!orig_cudaGetLastError) fprintf(stderr, "Failed to hook cudaGetLastError\n");
    
    // 初始化分配追踪
    alloc_capacity = 128;
    allocations = malloc(alloc_capacity * sizeof(Allocation));
    
    fprintf(stderr, "[%s] CUDA Hook initialized. Hooked %d functions.\n", 
            get_timestamp(), 
            (orig_cudaMalloc != NULL) + (orig_cudaFree != NULL) + 
            (orig_cudaMemcpy != NULL) + (orig_cudaMemset != NULL) +
            (orig_cudaLaunchKernel != NULL) + (orig_cudaDeviceSynchronize != NULL) +
            (orig_cudaGetLastError != NULL));
    fflush(stderr);
}

// 添加分配记录
static void add_allocation(void* ptr, size_t size) {
    if (alloc_count >= alloc_capacity) {
        alloc_capacity *= 2;
        allocations = realloc(allocations, alloc_capacity * sizeof(Allocation));
    }
    allocations[alloc_count].ptr = ptr;
    allocations[alloc_count].size = size;
    alloc_count++;
}

// 查找并移除分配记录
static size_t remove_allocation(void* ptr) {
    for (size_t i = 0; i < alloc_count; i++) {
        if (allocations[i].ptr == ptr) {
            size_t size = allocations[i].size;
            allocations[i] = allocations[alloc_count - 1];
            alloc_count--;
            return size;
        }
    }
    return 0; // 未找到分配记录
}

// 确保使用C链接规范
#ifdef __cplusplus
extern "C" {
#endif

// Hook 函数实现 - 使用更通用的声明
cudaError_t cudaMalloc(void **devPtr, size_t size) {
    if (!orig_cudaMalloc) init_orig_functions();
    if (!orig_cudaMalloc) return cudaErrorUnknown;
    
    cudaError_t err = orig_cudaMalloc(devPtr, size);
    
    if (err == cudaSuccess) {
        total_allocated += size;
        current_allocated += size;
        if (current_allocated > peak_allocated) {
            peak_allocated = current_allocated;
        }
        add_allocation(*devPtr, size);
        fprintf(stderr, "[%s] HOOK: cudaMalloc(ptr=%p, size=%zu) -> %s\n", 
               get_timestamp(), *devPtr, size, cudaGetErrorString(err));
    } else {
        fprintf(stderr, "[%s] HOOK: cudaMalloc(size=%zu) FAILED -> %s\n", 
               get_timestamp(), size, cudaGetErrorString(err));
    }
    fflush(stderr);
    
    return err;
}

cudaError_t cudaFree(void *devPtr) {
    if (!orig_cudaFree) init_orig_functions();
    if (!orig_cudaFree) return cudaErrorUnknown;
    
    size_t size = remove_allocation(devPtr);
    
    cudaError_t err = orig_cudaFree(devPtr);
    
    if (err == cudaSuccess) {
        current_allocated -= size;
        fprintf(stderr, "[%s] HOOK: cudaFree(ptr=%p, size=%zu) -> %s\n", 
               get_timestamp(), devPtr, size, cudaGetErrorString(err));
    } else {
        fprintf(stderr, "[%s] HOOK: cudaFree(ptr=%p) -> %s\n", 
               get_timestamp(), devPtr, cudaGetErrorString(err));
    }
    fflush(stderr);
    
    return err;
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
    if (!orig_cudaMemcpy) init_orig_functions();
    if (!orig_cudaMemcpy) return cudaErrorUnknown;
    
    cudaError_t err = orig_cudaMemcpy(dst, src, count, kind);
    
    if (err == cudaSuccess) {
        memcpy_bytes += count;
        const char *kind_str = "Unknown";
        switch (kind) {
            case cudaMemcpyHostToHost: kind_str = "HostToHost"; break;
            case cudaMemcpyHostToDevice: kind_str = "HostToDevice"; break;
            case cudaMemcpyDeviceToHost: kind_str = "DeviceToHost"; break;
            case cudaMemcpyDeviceToDevice: kind_str = "DeviceToDevice"; break;
        }
        fprintf(stderr, "[%s] HOOK: cudaMemcpy(dst=%p, src=%p, size=%zu, kind=%s) -> %s\n", 
               get_timestamp(), dst, src, count, kind_str, cudaGetErrorString(err));
    } else {
        fprintf(stderr, "[%s] HOOK: cudaMemcpy(size=%zu) FAILED -> %s\n", 
               get_timestamp(), count, cudaGetErrorString(err));
    }
    fflush(stderr);
    
    return err;
}

// 其他函数实现类似...

#ifdef __cplusplus
}
#endif

// 程序退出时打印统计信息
__attribute__((destructor))
static void print_summary() {
    fprintf(stderr, "\n===== CUDA HOOK SUMMARY =====\n");
    fprintf(stderr, "Total Allocated: %.2f MB\n", total_allocated / (1024.0 * 1024.0));
    fprintf(stderr, "Peak Allocated:  %.2f MB\n", peak_allocated / (1024.0 * 1024.0));
    fprintf(stderr, "Current Allocated: %.2f MB\n", current_allocated / (1024.0 * 1024.0));
    fprintf(stderr, "Total Memcpy:    %.2f MB\n", memcpy_bytes / (1024.0 * 1024.0));
    fprintf(stderr, "Kernel Launches: %u\n", kernel_launches);
    fprintf(stderr, "Unfreed allocations: %zu\n", alloc_count);
    fprintf(stderr, "=============================\n");
    fflush(stderr);
    
    // 清理分配记录
    free(allocations);
}