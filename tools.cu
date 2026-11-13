#include "tools.h"
#include "cuda.h"
#include <map>
#include <vector>
#include <string>
#include <cstdio>
#include <cstring>

__global__ void compare_pages_kernel(const uint4* a, const uint4* b, size_t size, unsigned int* result) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    bool different = false;
    for (size_t i = index; i < size / sizeof(uint4); i += stride) {
        uint4 x = a[i], y = b[i];
        different |= x.x != y.x || x.y != y.y || x.z != y.z || x.w != y.w;
    }
    for (size_t i = size / sizeof(uint4) * sizeof(uint4) + index; i < size; i += stride)
        different |= reinterpret_cast<const uint8_t*>(a)[i] != reinterpret_cast<const uint8_t*>(b)[i];
    if (__syncthreads_or(different) && threadIdx.x == 0)
        atomicExch(result, 1u);
}

bool gpu_pages_equal(unsigned long long a, unsigned long long b, size_t size) {
    struct Buffer {
        CUcontext context = nullptr;
        unsigned long long id = 0;
        unsigned int* result = nullptr;
        ~Buffer() {
            CUcontext previous;
            unsigned long long current_id;
            if (result && cuCtxGetId(context, &current_id) == CUDA_SUCCESS && current_id == id && cuCtxPushCurrent(context) == CUDA_SUCCESS) {
                cudaFree(result);
                cuCtxPopCurrent(&previous);
            }
        }
    };
    CUcontext context;
    unsigned long long id;
    if (cuCtxGetCurrent(&context) != CUDA_SUCCESS || cuCtxGetId(context, &id) != CUDA_SUCCESS) return false;
    thread_local std::map<unsigned long long, Buffer> buffers;
    auto& buffer = buffers[id];
    buffer.context = context;
    buffer.id = id;
    if (!buffer.result && cudaMalloc(&buffer.result, sizeof(unsigned int)) != cudaSuccess) return false;
    cudaStream_t stream = cudaStreamPerThread;
    if (cudaMemsetAsync(buffer.result, 0, sizeof(unsigned int), stream) != cudaSuccess) return false;
    compare_pages_kernel<<<128, 256, 0, stream>>>(reinterpret_cast<const uint4*>(a), reinterpret_cast<const uint4*>(b), size, buffer.result);
    if (cudaGetLastError() != cudaSuccess) return false;
    unsigned int different;
    if (cudaMemcpyAsync(&different, buffer.result, sizeof(different), cudaMemcpyDeviceToHost, stream) != cudaSuccess) return false;
    return cudaStreamSynchronize(stream) == cudaSuccess && different == 0;
}

std::string current_time() {
    char buffer[64];
    time_t rawtime;
    struct tm *timeinfo;
    
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", timeinfo);
    
    return std::string(buffer);
}

std::string format_bytes(size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit_index = 0;
    double size = static_cast<double>(bytes);
    
    while (size >= 1024 && unit_index < 4) {
        size /= 1024;
        unit_index++;
    }

    char buffer[32];
    snprintf(buffer, sizeof(buffer), "%.2f %s", size, units[unit_index]);
    return std::string(buffer);
}

////////////////////////////////////////////////
// sha256
////////////////////////////////////////////////
std::string sha256(const void* data, size_t size) {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, data, size);
    SHA256_Final(hash, &sha256);
    
    char buf[2 * SHA256_DIGEST_LENGTH + 1];
    buf[2 * SHA256_DIGEST_LENGTH] = 0;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; i++)
        sprintf(buf + i * 2, "%02x", hash[i]);
    
    return std::string(buf);
}

////////////////////////////////////////////////
// XXH64
////////////////////////////////////////////////
std::string fast_hash(const void* data, size_t size) {
    XXH64_hash_t hash = XXH3_64bits(data, size);
    std::ostringstream oss;
    oss << std::hex << std::setfill('0') << std::setw(16) << hash;
    return oss.str();
}

////////////////////////////////////////////////
// XXH64 gpu
////////////////////////////////////////////////

// 常量：哈希算法参数

__global__ void gpu_hash_kernel_single(
    const uint8_t* __restrict__ data,
    size_t size,
    uint64_t* result)
{
    // 每个线程处理256字节 (16个16字节块)
    const int tid = threadIdx.x;
    const int block_id = blockIdx.x;
    const int chunk_size = 16;
    const int chunks_per_thread = 16;  // 每个线程处理16个块
    
    // 计算当前线程处理的起始位置
    const uint64_t start_offset = static_cast<uint64_t>(block_id) * blockDim.x * chunks_per_thread * chunk_size
                                + static_cast<uint64_t>(tid) * chunks_per_thread * chunk_size;
    
    uint64_t acc = PRIME64_5;
    
    // 处理当前线程负责的16个数据块
    for (int i = 0; i < chunks_per_thread; i++) {
        const uint64_t base_offset = start_offset + i * chunk_size;
        
        // 边界检查：确保不会读取越界
        if (base_offset < size) {
            uint64_t v1 = 0;
            uint64_t v2 = 0;
            
            // 读取第一个64位值
            if (base_offset + 8 <= size) {
                v1 = *reinterpret_cast<const uint64_t*>(data + base_offset);
            }
            
            // 读取第二个64位值
            if (base_offset + 16 <= size) {
                v2 = *reinterpret_cast<const uint64_t*>(data + base_offset + 8);
            }
            
            // 避免零值影响哈希
            if (v1 == 0) v1 = 1;
            if (v2 == 0) v2 = 1;
            
            // 局部哈希计算
            uint64_t local_acc = PRIME64_5;
            local_acc += PRIME64_1 * v1;
            local_acc = (local_acc << 31) | (local_acc >> 33);  // 旋转
            local_acc *= PRIME64_2;
            local_acc += PRIME64_1 * v2;
            local_acc = (local_acc << 31) | (local_acc >> 33);  // 旋转
            local_acc *= PRIME64_2;
            
            // 合并到线程累加器
            acc ^= local_acc;
            acc = acc * PRIME64_1 + PRIME64_4;
        }
    }
    
    // 使用共享内存进行block内归约
    __shared__ uint64_t shared_acc[256];
    shared_acc[tid] = acc;
    __syncthreads();
    
    // 并行归约 (使用XOR)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_acc[tid] ^= shared_acc[tid + s];
        }
        __syncthreads();
    }
    
    // 保存最终结果
    if (tid == 0) {
        uint64_t final_acc = shared_acc[0];
        
        // 最终混合
        final_acc ^= final_acc >> 33;
        final_acc *= PRIME64_2;
        final_acc ^= final_acc >> 29;
        final_acc *= PRIME64_3;
        final_acc ^= final_acc >> 32;
        
        // 原子操作确保多个block正确累加
        atomicAdd(reinterpret_cast<unsigned long long*>(result), final_acc);
    }
}

std::string gpu_fast_hash_single(unsigned long long d_data, size_t size) {
    // 初始化XXH3 GPU内存池（只分配一次）
    if (!g_memory_initialized_xxh3) {
        cudaMalloc(reinterpret_cast<void**>(&g_d_result_xxh3), sizeof(uint64_t));
        g_memory_initialized_xxh3 = true;
    }
    
    // 初始化CUDA stream（只创建一次）
    if (!g_hash_stream_initialized) {
        cudaStreamCreate(&g_hash_stream);
        g_hash_stream_initialized = true;
    }
    
    // 需要cudaMemset，因为atomicAdd需要从0开始累加
    cudaMemsetAsync(g_d_result_xxh3, 0, sizeof(uint64_t), g_hash_stream);
    
    const int threads_per_block = 256;
    
    const size_t bytes_per_block = threads_per_block * 16 * 16;  // 每个block处理256*256=65,536字节
    const int blocks = (size + bytes_per_block - 1) / bytes_per_block;

    gpu_hash_kernel_single<<<blocks, threads_per_block, 0, g_hash_stream>>>(
        reinterpret_cast<const uint8_t*>(d_data), 
        size, 
        g_d_result_xxh3
    );

    cudaStreamSynchronize(g_hash_stream);
    
    uint64_t h_result = 0;
    cudaMemcpyAsync(&h_result, g_d_result_xxh3, sizeof(uint64_t), cudaMemcpyDeviceToHost, g_hash_stream);
    
    cudaStreamSynchronize(g_hash_stream);
    
    char hex_str[17];
    snprintf(hex_str, sizeof(hex_str), "%016lx", h_result);
    return std::string(hex_str, 16);
}

// 修改后的GPU哈希kernel
__global__ void gpu_hash_kernel_batch(
    unsigned long long* __restrict__ d_pages,  // 现在这是GPU上的地址数组
    size_t page_size,
    size_t num_pages,
    char* hash_strings)
{
    const int page_idx = blockIdx.x;
    if (page_idx >= num_pages) return;

    // 现在 d_pages 在 GPU 上，可以直接访问
    unsigned long long page_addr = d_pages[page_idx];
    const uint8_t* data = reinterpret_cast<const uint8_t*>(page_addr);
    
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    const int chunk_size = 16;
    const int chunks_per_thread = 16;  // 每批处理16个块
    
    extern __shared__ uint64_t s_acc[];
    __shared__ uint64_t s_batch_results[32];  // 存储每个批次的结果（最多32个批次）
    
    // 计算总的批次数（类似 single 模式的多个 block）
    const size_t bytes_per_batch = num_threads * chunks_per_thread * chunk_size;  // 256*16*16=65536
    const int num_batches = (page_size + bytes_per_batch - 1) / bytes_per_batch;
    
    if (tid == 0) {
        for (int i = 0; i < num_batches && i < 32; i++) {
            s_batch_results[i] = 0;
        }
    }
    __syncthreads();
    
    // 循环处理多个批次，模拟 single 模式的多个 block
    for (int batch = 0; batch < num_batches; batch++) {
        uint64_t acc = PRIME64_5;  // 每个批次独立的线程累加器
        
        // 计算当前批次中线程的起始位置
        const size_t batch_base = batch * bytes_per_batch;
        const size_t start_offset = batch_base + tid * chunks_per_thread * chunk_size;
        
        // 处理当前批次的16个数据块（与 single 模式每个 block 的逻辑一致）
        for (int i = 0; i < chunks_per_thread; i++) {
            const size_t base_offset = start_offset + i * chunk_size;
            
            // 边界检查：确保不会读取越界（与 single 模式完全一致）
            if (base_offset < page_size) {
                uint64_t v1 = 0;
                uint64_t v2 = 0;
                
                // 读取第一个64位值（与 single 模式完全一致）
                if (base_offset + 8 <= page_size) {
                    v1 = *reinterpret_cast<const uint64_t*>(data + base_offset);
                }
                
                // 读取第二个64位值（与 single 模式完全一致）
                if (base_offset + 16 <= page_size) {
                    v2 = *reinterpret_cast<const uint64_t*>(data + base_offset + 8);
                }
                
                // 避免零值影响哈希
                if (v1 == 0) v1 = 1;
                if (v2 == 0) v2 = 1;
                
                // 局部哈希计算（与 single 模式完全一致）
                uint64_t local_acc = PRIME64_5;
                local_acc += PRIME64_1 * v1;
                local_acc = (local_acc << 31) | (local_acc >> 33);
                local_acc *= PRIME64_2;
                local_acc += PRIME64_1 * v2;
                local_acc = (local_acc << 31) | (local_acc >> 33);
                local_acc *= PRIME64_2;
                
                // 合并到线程累加器（与 single 模式完全一致）
                acc ^= local_acc;
                acc = acc * PRIME64_1 + PRIME64_4;
            }
        }
        
        // 批次内归约（与 single 模式的 block 内归约一致）
        s_acc[tid] = acc;
        __syncthreads();
        
        for (int s = num_threads / 2; s > 0; s >>= 1) {
            if (tid < s) {
                s_acc[tid] ^= s_acc[tid + s];
            }
            __syncthreads();
        }
        
        // 对该批次做最终混合，并累加到批次结果（模拟 atomicAdd）
        if (tid == 0) {
            uint64_t batch_acc = s_acc[0];
            
            // 最终混合（与 single 模式完全一致）
            batch_acc ^= batch_acc >> 33;
            batch_acc *= PRIME64_2;
            batch_acc ^= batch_acc >> 29;
            batch_acc *= PRIME64_3;
            batch_acc ^= batch_acc >> 32;
            
            // 累加该批次结果（模拟 atomicAdd）
            s_batch_results[batch] = batch_acc;
        }
        __syncthreads();
    }
    
    // 汇总所有批次的结果（模拟 single 模式的多个 block 通过 atomicAdd 累加）
    if (tid == 0) {
        uint64_t final_acc = 0;
        for (int i = 0; i < num_batches; i++) {
            final_acc += s_batch_results[i];  // 直接相加，与 atomicAdd 效果一致
        }
        
        // 转换为十六进制字符串
        char* hash_str = hash_strings + page_idx * 16;
        const char hex_chars[] = "0123456789abcdef";
        
        for (int i = 0; i < 16; i++) {
            int nibble = (final_acc >> (60 - i * 4)) & 0xF;
            hash_str[i] = hex_chars[nibble];
        }
    }
}

std::vector<std::string> gpu_fast_hash_batch(
    unsigned long long* h_pages,  
    size_t page_size, 
    size_t num_pages) {
    
    // 初始化页面地址池（只分配一次）
    const size_t default_pages_count = 64;
    const size_t required_pages_size = num_pages * sizeof(unsigned long long);
    
    if (!g_d_pages_pool_initialized) {
        cudaMalloc(&g_d_pages_pool, default_pages_count * sizeof(unsigned long long));
        g_d_pages_pool_size = default_pages_count * sizeof(unsigned long long);
        g_d_pages_pool_initialized = true;
    }
    
    // 如果需要更多空间，则扩展内存池
    if (required_pages_size > g_d_pages_pool_size) {
        fprintf(stderr, "Expanding pages pool from %zu to %zu bytes\n", 
                g_d_pages_pool_size, required_pages_size);
        cudaFree(g_d_pages_pool);
        cudaMalloc(&g_d_pages_pool, required_pages_size);
        g_d_pages_pool_size = required_pages_size;
    }
    
    // 复用内存池，拷贝页面地址到设备
    cudaMemcpy(g_d_pages_pool, h_pages, required_pages_size, 
               cudaMemcpyHostToDevice);
    
    const size_t default_pool_size = 64 * 16;  // 64个页面的哈希字符串
    const size_t required_size = num_pages * 16;
    
    if (!g_hash_strings_pool_initialized) {
        cudaMalloc(&g_d_hash_strings_pool, default_pool_size);
        g_hash_strings_pool_size = default_pool_size;
        g_hash_strings_pool_initialized = true;
    }
    
    if (required_size > g_hash_strings_pool_size) {
        fprintf(stderr, "Expanding hash strings pool from %zu to %zu\n", 
                g_hash_strings_pool_size, required_size);
        cudaFree(g_d_hash_strings_pool);
        cudaMalloc(&g_d_hash_strings_pool, required_size);
        g_hash_strings_pool_size = required_size;
    }
    
    const int threads_per_block = 256;
    const int blocks = num_pages;  // 每个block处理一个页面
    const size_t shared_mem_size = threads_per_block * sizeof(uint64_t);
    
    // 确保使用全局stream
    if (!g_hash_stream_initialized) {
        cudaStreamCreate(&g_hash_stream);
        g_hash_stream_initialized = true;
    }

    gpu_hash_kernel_batch<<<blocks, threads_per_block, shared_mem_size, g_hash_stream>>>(
        g_d_pages_pool,           
        page_size, 
        num_pages, 
        g_d_hash_strings_pool
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("GPU kernel execution failed");
    }
    
    cudaStreamSynchronize(g_hash_stream);
    
    std::vector<char> h_hash_strings(required_size);
    cudaMemcpyAsync(h_hash_strings.data(), g_d_hash_strings_pool, required_size, 
               cudaMemcpyDeviceToHost, g_hash_stream);
    cudaStreamSynchronize(g_hash_stream);
    
    std::vector<std::string> results;
    results.reserve(num_pages);
    for (size_t i = 0; i < num_pages; i++) {
        results.emplace_back(h_hash_strings.data() + i * 16, 16);
    }
    
    return results;
}

////////////////////////////////////////////////
// 优化版 batch 哈希（每页一个 block，grid-stride + 合并访问）
////////////////////////////////////////////////

// 原实现保留；此为新增优化版。
__global__ void gpu_hash_kernel_batch_opt(
    const unsigned long long* __restrict__ d_pages,
    size_t page_size,
    size_t num_pages,
    char* __restrict__ hash_strings)
{
    const int page_idx = blockIdx.x;
    if (page_idx >= num_pages) return;

    const uint8_t* data = reinterpret_cast<const uint8_t*>(d_pages[page_idx]);

    const uint64_t chunk_size = 16ULL;
    const int num_threads = blockDim.x;
    const int chunks_per_thread = 16; // 与原版保持一致
    const uint64_t bytes_per_batch = (uint64_t)num_threads * (uint64_t)chunks_per_thread * chunk_size;
    const int num_batches = (int)((page_size + bytes_per_batch - 1ULL) / bytes_per_batch);
    extern __shared__ uint64_t s_acc[]; // 动态共享内存，与原版布局一致
    __shared__ uint64_t s_batch_results[32];  // 静态共享内存，紧随 s_acc 之后
    if (threadIdx.x == 0) {
        for (int i = 0; i < num_batches && i < 32; i++) {
            s_batch_results[i] = 0ULL;
        }
    }
    __syncthreads();

    // 为每个批次独立计算与归约，严格复现：每批次先归约->最终混合->相加
    for (int batch = 0; batch < num_batches; ++batch) {
        const uint64_t batch_base = (uint64_t)batch * bytes_per_batch;
        const uint64_t start_offset = batch_base + (uint64_t)threadIdx.x * (uint64_t)chunks_per_thread * chunk_size;

        uint64_t acc = PRIME64_5;

        // 与原版严格一致：每线程处理固定的16个块，顺序相同
        for (int i = 0; i < chunks_per_thread; ++i) {
            const uint64_t base = start_offset + (uint64_t)i * chunk_size;
            if (base >= page_size) break;

            uint64_t v1 = 0ULL, v2 = 0ULL;
            if (base + 8ULL <= page_size) {
                v1 = *reinterpret_cast<const uint64_t*>(data + base);
            }
            if (base + 16ULL <= page_size) {
                v2 = *reinterpret_cast<const uint64_t*>(data + base + 8ULL);
            }

            if (v1 == 0ULL) v1 = 1ULL;
            if (v2 == 0ULL) v2 = 1ULL;

            uint64_t local_acc = PRIME64_5;
            local_acc += PRIME64_1 * v1;
            local_acc = (local_acc << 31) | (local_acc >> 33);
            local_acc *= PRIME64_2;
            local_acc += PRIME64_1 * v2;
            local_acc = (local_acc << 31) | (local_acc >> 33);
            local_acc *= PRIME64_2;

            acc ^= local_acc;
            acc = acc * PRIME64_1 + PRIME64_4;
        }

        // 批次内归约（与原版一致：共享内存 + XOR 归约）
        s_acc[threadIdx.x] = acc;
        __syncthreads();
        for (int s = num_threads / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                s_acc[threadIdx.x] ^= s_acc[threadIdx.x + s];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            // 对本批次的归约结果做最终混合
            uint64_t final_acc = s_acc[0];
            final_acc ^= final_acc >> 33;
            final_acc *= PRIME64_2;
            final_acc ^= final_acc >> 29;
            final_acc *= PRIME64_3;
            final_acc ^= final_acc >> 32;
            // 与原实现一致：不同批次的 final_acc 写入共享数组（可能越界的原行为由上层保持一致性）
            s_batch_results[batch] = final_acc;
        }
        __syncthreads();
    }

    // 写出最终结果（为所有批次混合后之和）——与原版一致：按 num_batches 遍历
    if (threadIdx.x == 0) {
        uint64_t final_sum = 0ULL;
        for (int i = 0; i < num_batches; ++i) final_sum += s_batch_results[i];
        char* hash_str = hash_strings + page_idx * 16;
        const char hex_chars[] = "0123456789abcdef";
        for (int i = 0; i < 16; i++) {
            int nibble = (final_sum >> (60 - i * 4)) & 0xF;
            hash_str[i] = hex_chars[nibble];
        }
    }
}

std::vector<std::string> gpu_fast_hash_batch_opt(
    unsigned long long* h_pages,
    size_t page_size,
    size_t num_pages)
{
    // 复用/初始化页面地址池
    const size_t required_pages_size = num_pages * sizeof(unsigned long long);
    if (!g_d_pages_pool_initialized) {
        cudaMalloc(&g_d_pages_pool, required_pages_size > 0 ? required_pages_size : sizeof(unsigned long long));
        g_d_pages_pool_size = required_pages_size > 0 ? required_pages_size : sizeof(unsigned long long);
        g_d_pages_pool_initialized = true;
    }
    if (required_pages_size > g_d_pages_pool_size) {
        cudaFree(g_d_pages_pool);
        cudaMalloc(&g_d_pages_pool, required_pages_size);
        g_d_pages_pool_size = required_pages_size;
    }

    if (required_pages_size) {
        cudaMemcpy(g_d_pages_pool, h_pages, required_pages_size, cudaMemcpyHostToDevice);
    }

    // 复用/初始化哈希字符串池
    const size_t required_size = num_pages * 16;
    if (!g_hash_strings_pool_initialized) {
        cudaMalloc(&g_d_hash_strings_pool, required_size > 0 ? required_size : 16);
        g_hash_strings_pool_size = required_size > 0 ? required_size : 16;
        g_hash_strings_pool_initialized = true;
    }
    if (required_size > g_hash_strings_pool_size) {
        cudaFree(g_d_hash_strings_pool);
        cudaMalloc(&g_d_hash_strings_pool, required_size);
        g_hash_strings_pool_size = required_size;
    }

    if (!g_hash_stream_initialized) {
        cudaStreamCreate(&g_hash_stream);
        g_hash_stream_initialized = true;
    }

    const int threads_per_block = 256;
    const int blocks = static_cast<int>(num_pages);

    const size_t shared_mem_size = threads_per_block * sizeof(uint64_t);
    gpu_hash_kernel_batch_opt<<<blocks, threads_per_block, shared_mem_size, g_hash_stream>>>(
        g_d_pages_pool,
        page_size,
        num_pages,
        g_d_hash_strings_pool
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_hash_kernel_batch_opt launch failed: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("gpu_hash_kernel_batch_opt failed");
    }

    cudaStreamSynchronize(g_hash_stream);

    std::vector<char> h_hash_strings(required_size);
    if (required_size) {
        cudaMemcpyAsync(h_hash_strings.data(), g_d_hash_strings_pool, required_size, cudaMemcpyDeviceToHost, g_hash_stream);
    }
    cudaStreamSynchronize(g_hash_stream);

    std::vector<std::string> results;
    results.reserve(num_pages);
    for (size_t i = 0; i < num_pages; i++) {
        results.emplace_back(h_hash_strings.data() + i * 16, 16);
    }
    return results;
}

////////////////////////////////////////////////
// 优化版 batch 哈希 v2（共享内存分块预取 + 原序计算）
////////////////////////////////////////////////

__global__ void gpu_hash_kernel_batch_opt2(
    const unsigned long long* __restrict__ d_pages,
    size_t page_size,
    size_t num_pages,
    char* __restrict__ hash_strings)
{
    const int page_idx = blockIdx.x;
    if (page_idx >= num_pages) return;

    const uint8_t* data = reinterpret_cast<const uint8_t*>(d_pages[page_idx]);

    const uint64_t chunk_size = 16ULL;
    const int num_threads = blockDim.x;
    const int chunks_per_thread = 16; // 与原版保持一致
    const uint64_t bytes_per_batch = (uint64_t)num_threads * (uint64_t)chunks_per_thread * chunk_size; // 64KB @256
    const int num_batches = (int)((page_size + bytes_per_batch - 1ULL) / bytes_per_batch);

    // 动态共享内存布局：tile 缓冲 + s_acc
    extern __shared__ unsigned char s_mem[];
    // 选择 16KB tile，避免 48KB 上限（必要时可调大为 32KB/48KB）
    const uint64_t kTileBytes = 16ULL * 1024ULL;
    unsigned char* s_tile = s_mem;
    uint64_t* s_acc = reinterpret_cast<uint64_t*>(s_mem + kTileBytes);

    __shared__ uint64_t s_batch_results[32];
    if (threadIdx.x == 0) {
        for (int i = 0; i < num_batches && i < 32; i++) s_batch_results[i] = 0ULL;
    }
    __syncthreads();

    for (int batch = 0; batch < num_batches; ++batch) {
        const uint64_t batch_base = (uint64_t)batch * bytes_per_batch;
        const uint64_t batch_end = min(batch_base + bytes_per_batch, (uint64_t)page_size);

        uint64_t acc = PRIME64_5;

        // 分 tile 预取到共享内存，每个 tile 最大 16KB，保证全局读合并
        for (uint64_t tile_base = batch_base; tile_base < batch_end; tile_base += kTileBytes) {
            const uint64_t tile_end = min(tile_base + kTileBytes, batch_end);
            const uint64_t tile_bytes = tile_end - tile_base;
            const uint64_t tile_chunks = (tile_bytes + chunk_size - 1ULL) / chunk_size; // 以 16B 为单位

            // 协作加载 tile 到共享内存（16B 对齐单元）
            for (uint64_t tchunk = threadIdx.x; tchunk < tile_chunks; tchunk += blockDim.x) {
                const uint64_t g_base = tile_base + tchunk * chunk_size;
                unsigned long long vv[2] = {0ULL, 0ULL};
                if (g_base + 16ULL <= page_size) {
                    memcpy(&vv[0], data + g_base, 16);
                } else {
                    if (g_base + 8ULL <= page_size) memcpy(&vv[0], data + g_base, 8);
                    if (g_base + 16ULL <= page_size) memcpy(&vv[1], data + g_base + 8ULL, 8);
                }
                // 写入共享 tile（连续 16B）
                const uint64_t soff = tchunk * chunk_size;
                memcpy(s_tile + soff, &vv[0], 16);
            }
            __syncthreads();

            // 严格原序：每个线程处理其 16 个块，若落在当前 tile 则立即计算
            const uint64_t thread_base = batch_base + (uint64_t)threadIdx.x * (uint64_t)chunks_per_thread * chunk_size;
            for (int i = 0; i < chunks_per_thread; ++i) {
                const uint64_t base = thread_base + (uint64_t)i * chunk_size;
                if (base >= batch_end) break;
                if (!(base >= tile_base && base + 16ULL <= tile_end)) continue; // 不在本 tile 内，等下一个 tile

                const uint64_t soff = (base - tile_base);
                unsigned long long vv[2];
                memcpy(&vv[0], s_tile + soff, 16);
                uint64_t v1 = vv[0];
                uint64_t v2 = vv[1];
                if (v1 == 0ULL) v1 = 1ULL;
                if (v2 == 0ULL) v2 = 1ULL;

                uint64_t local_acc = PRIME64_5;
                local_acc += PRIME64_1 * v1;
                local_acc = (local_acc << 31) | (local_acc >> 33);
                local_acc *= PRIME64_2;
                local_acc += PRIME64_1 * v2;
                local_acc = (local_acc << 31) | (local_acc >> 33);
                local_acc *= PRIME64_2;

                acc ^= local_acc;
                acc = acc * PRIME64_1 + PRIME64_4;
            }
            __syncthreads();
        }

        // 批次内归约（与原版一致）
        s_acc[threadIdx.x] = acc;
        __syncthreads();
        for (int s = num_threads / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) s_acc[threadIdx.x] ^= s_acc[threadIdx.x + s];
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            uint64_t final_acc = s_acc[0];
            final_acc ^= final_acc >> 33;
            final_acc *= PRIME64_2;
            final_acc ^= final_acc >> 29;
            final_acc *= PRIME64_3;
            final_acc ^= final_acc >> 32;
            s_batch_results[batch] = final_acc;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        uint64_t final_sum = 0ULL;
        for (int i = 0; i < num_batches; ++i) final_sum += s_batch_results[i];
        char* hash_str = hash_strings + page_idx * 16;
        const char hex_chars[] = "0123456789abcdef";
        for (int i = 0; i < 16; i++) {
            int nibble = (final_sum >> (60 - i * 4)) & 0xF;
            hash_str[i] = hex_chars[nibble];
        }
    }
}

std::vector<std::string> gpu_fast_hash_batch_opt2(
    unsigned long long* h_pages,
    size_t page_size,
    size_t num_pages)
{
    const size_t required_pages_size = num_pages * sizeof(unsigned long long);
    if (!g_d_pages_pool_initialized) {
        cudaMalloc(&g_d_pages_pool, required_pages_size > 0 ? required_pages_size : sizeof(unsigned long long));
        g_d_pages_pool_size = required_pages_size > 0 ? required_pages_size : sizeof(unsigned long long);
        g_d_pages_pool_initialized = true;
    }
    if (required_pages_size > g_d_pages_pool_size) {
        cudaFree(g_d_pages_pool);
        cudaMalloc(&g_d_pages_pool, required_pages_size);
        g_d_pages_pool_size = required_pages_size;
    }
    if (required_pages_size) {
        cudaMemcpy(g_d_pages_pool, h_pages, required_pages_size, cudaMemcpyHostToDevice);
    }

    const size_t required_size = num_pages * 16;
    if (!g_hash_strings_pool_initialized) {
        cudaMalloc(&g_d_hash_strings_pool, required_size > 0 ? required_size : 16);
        g_hash_strings_pool_size = required_size > 0 ? required_size : 16;
        g_hash_strings_pool_initialized = true;
    }
    if (required_size > g_hash_strings_pool_size) {
        cudaFree(g_d_hash_strings_pool);
        cudaMalloc(&g_d_hash_strings_pool, required_size);
        g_hash_strings_pool_size = required_size;
    }

    if (!g_hash_stream_initialized) {
        cudaStreamCreate(&g_hash_stream);
        g_hash_stream_initialized = true;
    }

    const int threads_per_block = 256;
    const int blocks = static_cast<int>(num_pages);
    const size_t kTileBytes = 16 * 1024; // 必须与 kernel 中保持一致
    const size_t shared_mem_size = kTileBytes + threads_per_block * sizeof(uint64_t);

    gpu_hash_kernel_batch_opt2<<<blocks, threads_per_block, shared_mem_size, g_hash_stream>>>(
        g_d_pages_pool,
        page_size,
        num_pages,
        g_d_hash_strings_pool
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_hash_kernel_batch_opt2 launch failed: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("gpu_hash_kernel_batch_opt2 failed");
    }

    cudaStreamSynchronize(g_hash_stream);

    std::vector<char> h_hash_strings(required_size);
    if (required_size) {
        cudaMemcpyAsync(h_hash_strings.data(), g_d_hash_strings_pool, required_size, cudaMemcpyDeviceToHost, g_hash_stream);
    }
    cudaStreamSynchronize(g_hash_stream);

    std::vector<std::string> results;
    results.reserve(num_pages);
    for (size_t i = 0; i < num_pages; i++) {
        results.emplace_back(h_hash_strings.data() + i * 16, 16);
    }
    return results;
}

////////////////////////////////////////////////
// 超快速哈希函数 - 用于预过滤
////////////////////////////////////////////////

// GPU版本的超快速哈希
__global__ void gpu_ultra_fast_hash_kernel(
    const uint8_t* __restrict__ data,
    size_t size,
    uint64_t* result)
{
    // const int tid = threadIdx.x;
    // const int block_id = blockIdx.x;
    // const int chunk_size = 8;
    // const int chunks_per_thread = 16;  // 减少到16个块，与XXH64一致
    
    // const uint64_t start_offset = static_cast<uint64_t>(block_id) * blockDim.x * chunks_per_thread * chunk_size
    //                             + static_cast<uint64_t>(tid) * chunks_per_thread * chunk_size;
    
    // uint64_t hash = 0x9E3779B97F4A7C15ULL;
    
    // for (int i = 0; i < chunks_per_thread; i++) {
    //     const uint64_t base_offset = start_offset + i * chunk_size;
        
    //     if (base_offset + 8 <= size) {
    //         uint64_t chunk = *reinterpret_cast<const uint64_t*>(data + base_offset);
    //         hash ^= chunk;
    //         hash = hash * 0x9E3779B97F4A7C15ULL;  // 简化：只做乘法，不做旋转
    //     }
    // }
    
    // // 使用共享内存进行block内归约
    // __shared__ uint64_t shared_hash[256];
    // shared_hash[tid] = hash;
    // __syncthreads();
    
    // // 并行归约 (使用XOR)
    // for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    //     if (tid < s) {
    //         shared_hash[tid] ^= shared_hash[tid + s];
    //     }
    //     __syncthreads();
    // }
    
    // if (tid == 0) {
    //     atomicXor(reinterpret_cast<unsigned long long*>(result), shared_hash[0]);
    // }
}

uint64_t gpu_ultra_fast_hash_single(unsigned long long d_data, size_t size) {
    // 初始化GPU内存池（只分配一次）
    if (!g_memory_initialized) {
        cudaMalloc(reinterpret_cast<void**>(&g_d_result), sizeof(uint64_t));
        g_memory_initialized = true;
    }
    
    // 确保使用全局stream
    if (!g_hash_stream_initialized) {
        cudaStreamCreate(&g_hash_stream);
        g_hash_stream_initialized = true;
    }
    
    const int threads_per_block = 256;
    const size_t bytes_per_block = threads_per_block * 16 * 8;  // 每个block处理256*128=32,768字节
    const int blocks = (size + bytes_per_block - 1) / bytes_per_block;

    gpu_ultra_fast_hash_kernel<<<blocks, threads_per_block, 0, g_hash_stream>>>(
        reinterpret_cast<const uint8_t*>(d_data), 
        size, 
        g_d_result
    );
    
    cudaStreamSynchronize(g_hash_stream);
    
    uint64_t h_result = 0;
    cudaMemcpyAsync(&h_result, g_d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost, g_hash_stream);
    cudaStreamSynchronize(g_hash_stream);
    
    return h_result;
}

////////////////////////////////////////////////
// 布隆过滤器实现
////////////////////////////////////////////////

// 生成多个哈希值用于布隆过滤器
void generate_bloom_hashes(uint64_t input_hash, uint64_t* hashes, int count) {
    uint64_t h1 = input_hash;
    uint64_t h2 = input_hash * 0x9E3779B97F4A7C15ULL;
    
    for (int i = 0; i < count; i++) {
        hashes[i] = (h1 + i * h2) % (BLOOM_FILTER_SIZE * 8);
    }
}

// 检查布隆过滤器是否可能包含该哈希
bool bloom_filter_might_contain(const unsigned char* bloom_filter, uint64_t fast_hash) {
    uint64_t hashes[BLOOM_FILTER_HASH_COUNT];
    generate_bloom_hashes(fast_hash, hashes, BLOOM_FILTER_HASH_COUNT);
    
    for (int i = 0; i < BLOOM_FILTER_HASH_COUNT; i++) {
        uint64_t bit_index = hashes[i];
        uint64_t byte_index = bit_index / 8;
        uint64_t bit_offset = bit_index % 8;
        
        if (byte_index >= BLOOM_FILTER_SIZE) continue;
        
        if (!(bloom_filter[byte_index] & (1 << bit_offset))) {
            return false; // 如果任何一个位为0，则肯定不存在
        }
    }
    
    return true; // 所有位都为1，可能存在
}

// 向布隆过滤器添加哈希
void bloom_filter_add(unsigned char* bloom_filter, uint64_t fast_hash) {
    uint64_t hashes[BLOOM_FILTER_HASH_COUNT];
    generate_bloom_hashes(fast_hash, hashes, BLOOM_FILTER_HASH_COUNT);
    
    for (int i = 0; i < BLOOM_FILTER_HASH_COUNT; i++) {
        uint64_t bit_index = hashes[i];
        uint64_t byte_index = bit_index / 8;
        uint64_t bit_offset = bit_index % 8;
        
        if (byte_index < BLOOM_FILTER_SIZE) {
            bloom_filter[byte_index] |= (1 << bit_offset);
        }
    }
}

std::string trim(const std::string& str) {
    size_t start = str.find_first_not_of(" \t\n\r");
    size_t end = str.find_last_not_of(" \t\n\r");
    if (start == std::string::npos || end == std::string::npos)
        return "";
    return str.substr(start, end - start + 1);
}

std::string to_lower(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

bool parse_bool(const std::string& value) {
    std::string lower_val = to_lower(trim(value));
    return (lower_val == "true" || lower_val == "yes" || lower_val == "1" || lower_val == "on");
}

int parse_int(const std::string& value, int default_value = 0) {
    try {
        return std::stoi(trim(value));
    } catch (...) {
        return default_value;
    }
}

size_t parse_size(const std::string& value, size_t default_value = 0) {
    try {
        std::string val = trim(value);
        size_t multiplier = 1;
        
        if (!val.empty()) {
            char last_char = val[val.size() - 1];
            if (last_char == 'K' || last_char == 'k') {
                multiplier = 1024;
                val = val.substr(0, val.size() - 1);
            } else if (last_char == 'M' || last_char == 'm') {
                multiplier = 1024 * 1024;
                val = val.substr(0, val.size() - 1);
            } else if (last_char == 'G' || last_char == 'g') {
                multiplier = 1024 * 1024 * 1024;
                val = val.substr(0, val.size() - 1);
            }
        }
        
        return std::stoull(val) * multiplier;
    } catch (...) {
        return default_value;
    }
}

bool parse_config_file(const std::string& filename, HookConfig& config) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        if (config.verbose_logging) {
            printf("Config file %s not found, using defaults\n", filename.c_str());
        }
        return false;
    }
    
    std::string line;
    int line_num = 0;

    while (std::getline(file, line)) {
        line_num++;
        size_t comment_pos = line.find('#');
        if (comment_pos != std::string::npos) {
            line = line.substr(0, comment_pos);
        }
        
        std::string trimmed_line = trim(line);
        if (trimmed_line.empty()) {
            continue;
        }

        size_t equals_pos = trimmed_line.find('=');
        if (equals_pos == std::string::npos) {
            if (config.verbose_logging) {
                printf("Invalid config line %d: %s\n", line_num, trimmed_line.c_str());
            }
            continue;
        }
        
        std::string key = trim(trimmed_line.substr(0, equals_pos));
        std::string value = trim(trimmed_line.substr(equals_pos + 1));
        
        if (value.empty()) {
            if (config.verbose_logging) {
                printf("Empty value for key %s at line %d\n", key.c_str(), line_num);
            }
            continue;
        }
        
        if (key == "enable_deduplication") {
            config.enable_deduplication = parse_bool(value);
        } else if (key == "enable_background_scan") {
            config.enable_background_scan = parse_bool(value);
        } else if (key == "initial_scan_interval_ms") {
            config.initial_scan_interval_ms = parse_int(value, 1000);
        } else if (key == "max_scan_interval_ms") {
            config.max_scan_interval_ms = parse_int(value, 1000);
        } else if (key == "boot_scan_interval_ms") {
            config.boot_scan_interval_ms = parse_int(value, 1000);
        } else if (key == "stable_threshold") {
            config.stable_threshold = parse_int(value, 2);
        } else if (key == "max_batch_size") {
            config.max_batch_size = parse_int(value, 64);
        } else if (key == "verbose_logging") {
            config.verbose_logging = parse_bool(value);
        } else if (key == "enable_statistics") {
            config.enable_statistics = parse_bool(value);
        } else if (key == "userfault_fd_path") {
            config.userfault_fd_path = strdup(value.c_str());
        } else if (key == "device_id") {
            config.device_id = parse_int(value, 0);
        } else if (key == "page_size") {
            config.page_size = parse_size(value, 0);
        } else if (key == "shared_memory_name") {
            config.shared_memory_name = strdup(value.c_str());
        } else if (key == "max_dedupe_pages") {
            config.max_dedupe_pages = parse_int(value, 1024);
        } else if (key == "max_shared_refs") {
            config.max_shared_refs = parse_int(value, 32);
        } else if (key == "max_scan_pages") {
            config.max_scan_pages = parse_int(value, 0);
        } else if (key == "hook_path") {
            config.hook_path = strdup(value.c_str());
        } else if (key == "orig_path") {
            config.orig_path = strdup(value.c_str());
        } else if (config.verbose_logging) {
            printf("Unknown config key: %s\n", key.c_str());
        } 
    }
    
    file.close();
    printf("Configuration loaded from %s\n", filename.c_str());
    return true;
}

void print_config(HookConfig g_hook_config) {    
    printf("=== CUDA Hook Configuration ===\n");
    printf("Deduplication: %s\n", g_hook_config.enable_deduplication ? "Enabled" : "Disabled");
    printf("Background Scan: %s\n", g_hook_config.enable_background_scan ? "Enabled" : "Disabled");
    printf("Init Scan Interval: %d ms\n", g_hook_config.initial_scan_interval_ms);
    printf("Max Scan Interval: %d ms\n", g_hook_config.max_scan_interval_ms);
    printf("Boot Scan Interval: %d ms\n", g_hook_config.boot_scan_interval_ms);
    printf("Stable Threshold: %d\n", g_hook_config.stable_threshold);
    printf("Max Batch Size: %d\n", g_hook_config.max_batch_size);
    printf("Verbose Logging: %s\n", g_hook_config.verbose_logging ? "Enabled" : "Disabled");
    printf("Statistics: %s\n", g_hook_config.enable_statistics ? "Enabled" : "Disabled");
    printf("Userfault FD Path: %s\n", g_hook_config.userfault_fd_path);
    printf("Device ID: %d\n", g_hook_config.device_id);
    printf("Page Size: %s\n", format_bytes(g_hook_config.page_size).c_str());
    printf("Shared Memory Name: %s\n", g_hook_config.shared_memory_name);
    printf("Max Dedupe Pages: %d\n", g_hook_config.max_dedupe_pages);
    printf("Max Shared Refs: %d\n", g_hook_config.max_shared_refs);
    printf("================================\n");
}

////////////////////////////////////////////////
// 优化版 XXH64 gpu（grid-stride + 合并访问）
////////////////////////////////////////////////

// 原始实现保留不动；此为新增优化版本
__global__ void gpu_hash_kernel_single_opt(
    const uint8_t* __restrict__ data,
    size_t size,
    uint64_t* result)
{
    const uint64_t chunk_size = 16ULL;
    const uint64_t num_chunks = (size + chunk_size - 1ULL) / chunk_size;

    uint64_t acc = PRIME64_5;

    // grid-stride 循环，确保 warp 内相邻线程访问相邻块，提升合并度
    for (uint64_t chunk_idx = blockIdx.x * blockDim.x + threadIdx.x;
         chunk_idx < num_chunks;
         chunk_idx += (uint64_t)gridDim.x * blockDim.x)
    {
        const uint64_t base = chunk_idx * chunk_size;

        uint64_t v1 = 0ULL, v2 = 0ULL;

        if (base + 16ULL <= size) {
            // 使用 memcpy 规避潜在未对齐带来的性能/可移植性问题
            unsigned long long vv[2];
            memcpy(&vv[0], data + base, 16);
            v1 = vv[0];
            v2 = vv[1];
        } else {
            if (base + 8ULL <= size) memcpy(&v1, data + base, 8);
            if (base + 16ULL <= size) memcpy(&v2, data + base + 8ULL, 8);
        }

        if (v1 == 0ULL) v1 = 1ULL;
        if (v2 == 0ULL) v2 = 1ULL;

        uint64_t local_acc = PRIME64_5;
        local_acc += PRIME64_1 * v1;
        local_acc = (local_acc << 31) | (local_acc >> 33);
        local_acc *= PRIME64_2;
        local_acc += PRIME64_1 * v2;
        local_acc = (local_acc << 31) | (local_acc >> 33);
        local_acc *= PRIME64_2;

        acc ^= local_acc;
        acc = acc * PRIME64_1 + PRIME64_4;
    }

    // block 内归约（XOR）。先做 warp 内归约，再由每个 warp 的 lane0 写入共享内存
    unsigned mask = 0xffffffffu;
    // warp 归约
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc ^= __shfl_xor_sync(mask, acc, offset);
    }

    __shared__ uint64_t warp_acc[32];
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5; // /32
    if (lane_id == 0) {
        warp_acc[warp_id] = acc;
    }
    __syncthreads();

    // 由 warp0 再次归约所有 warp 的结果
    uint64_t block_acc = 0ULL;
    if (warp_id == 0) {
        block_acc = (lane_id < (blockDim.x + 31) / 32) ? warp_acc[lane_id] : 0ULL;
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_acc ^= __shfl_xor_sync(mask, block_acc, offset);
        }
        if (lane_id == 0) {
            uint64_t final_acc = block_acc;
            final_acc ^= final_acc >> 33;
            final_acc *= PRIME64_2;
            final_acc ^= final_acc >> 29;
            final_acc *= PRIME64_3;
            final_acc ^= final_acc >> 32;
            atomicAdd(reinterpret_cast<unsigned long long*>(result), final_acc);
        }
    }
}

// 优化版 host 包装：保留与原版相同的全局资源与返回格式
std::string gpu_fast_hash_single_opt(unsigned long long d_data, size_t size) {
    if (!g_memory_initialized_xxh3) {
        cudaMalloc(reinterpret_cast<void**>(&g_d_result_xxh3), sizeof(uint64_t));
        g_memory_initialized_xxh3 = true;
    }

    if (!g_hash_stream_initialized) {
        cudaStreamCreate(&g_hash_stream);
        g_hash_stream_initialized = true;
    }

    cudaMemsetAsync(g_d_result_xxh3, 0, sizeof(uint64_t), g_hash_stream);

    const int threads_per_block = 256;
    const uint64_t chunk_size = 16ULL;
    const uint64_t num_chunks = (size + chunk_size - 1ULL) / chunk_size;

    // grid 大小基于 chunks，过大输入时可自动扩展；也可按需做上限
    uint64_t blocks64 = (num_chunks + threads_per_block - 1ULL) / threads_per_block;
    int blocks = (blocks64 > 0x7fffffffULL) ? 0x7fffffff : static_cast<int>(blocks64);
    if (blocks <= 0) blocks = 1;

    gpu_hash_kernel_single_opt<<<blocks, threads_per_block, 0, g_hash_stream>>>(
        reinterpret_cast<const uint8_t*>(d_data),
        size,
        g_d_result_xxh3
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_hash_kernel_single_opt launch failed: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("gpu_hash_kernel_single_opt failed");
    }

    cudaStreamSynchronize(g_hash_stream);

    uint64_t h_result = 0;
    cudaMemcpyAsync(&h_result, g_d_result_xxh3, sizeof(uint64_t), cudaMemcpyDeviceToHost, g_hash_stream);
    cudaStreamSynchronize(g_hash_stream);

    char hex_str[17];
    snprintf(hex_str, sizeof(hex_str), "%016lx", h_result);
    return std::string(hex_str, 16);
}