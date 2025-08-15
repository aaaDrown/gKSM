#include "tools.h"

std::string current_time() {
    char buffer[64];
    time_t rawtime;
    struct tm *timeinfo;
    
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", timeinfo);
    
    return std::string(buffer);
}

void get_command_line() {
    char path[256];
    char cmdline[4096];
    int i;

    // 构建路径 /proc/self/cmdline
    snprintf(path, sizeof(path), "/proc/self/cmdline");

    // 打开并读取文件
    FILE *fp = fopen(path, "r");
    if (fp == NULL) {
        perror("fopen");
        return;
    }

    if (fgets(cmdline, sizeof(cmdline), fp) != NULL) {
        // 将空字符替换为换行符以便打印
        for (i = 0; i < (int)strlen(cmdline); i++) {
            if (cmdline[i] == '\0') {
                cmdline[i] = '\n';
            }
        }
        printf("Command Line:\n%s\n", cmdline);
    } else {
        perror("fgets");
    }

    fclose(fp);
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

// GPU内核：批量页面哈希计算
__global__ void gpu_hash_kernel_batch(
    const uint8_t* __restrict__ pages,
    size_t page_size,
    size_t num_pages,
    uint64_t* results)
{
    const int page_idx = blockIdx.x;
    if (page_idx >= num_pages) return;
    
    const uint8_t* data = pages + page_idx * page_size;
    const int tid = threadIdx.x;
    const int chunk_size = 16;
    const uint64_t base_offset = tid * chunk_size;
    
    uint64_t acc = PRIME64_5;

    if (base_offset < page_size) {
        const uint64_t* ptr = reinterpret_cast<const uint64_t*>(data + base_offset);
        const uint64_t v1 = ptr[0];
        const uint64_t v2 = ptr[1];

        acc += PRIME64_1 * v1;
        acc = ((acc << 31) | (acc >> 33)) * PRIME64_2;
        acc += PRIME64_1 * v2;
        acc = ((acc << 31) | (acc >> 33)) * PRIME64_2;
    }

    // Warp级归约
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc ^= __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    // Block级归约
    __shared__ uint64_t warp_results[32];
    if (tid % 32 == 0) {
        warp_results[tid / 32] = acc;
    }
    __syncthreads();

    if (tid == 0) {
        uint64_t final_acc = PRIME64_5;
        for (int i = 0; i < blockDim.x / 32; i++) {
            final_acc ^= warp_results[i];
            final_acc = ((final_acc << 31) | (final_acc >> 33)) * PRIME64_1;
        }
        
        final_acc ^= final_acc >> 33;
        final_acc *= PRIME64_2;
        final_acc ^= final_acc >> 29;
        final_acc *= PRIME64_3;
        final_acc ^= final_acc >> 32;
        
        results[page_idx] = final_acc;
    }
}

// 主机函数：单个页面哈希
std::string gpu_fast_hash_single(unsigned long long d_data, size_t size) {

    uint64_t* d_result = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_result), sizeof(uint64_t));
    cudaMemset(d_result, 0, sizeof(uint64_t));
    
    const int threads_per_block = 256;
    
    const size_t bytes_per_block = threads_per_block * 16 * 16;  // 每个block处理256*256=65,536字节
    const int blocks = (size + bytes_per_block - 1) / bytes_per_block;

    gpu_hash_kernel_single<<<blocks, threads_per_block>>>(
        reinterpret_cast<const uint8_t*>(d_data), 
        size, 
        d_result
    );
    
    cudaDeviceSynchronize();
    
    uint64_t h_result = 0;
    cudaMemcpy(&h_result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    
    char hex_str[17];
    snprintf(hex_str, sizeof(hex_str), "%016lx", h_result);
    return std::string(hex_str, 16);
}

// 主机函数：批量页面哈希
void gpu_fast_hash_batch(
    const void** pages,
    size_t page_size,
    size_t num_pages,
    uint64_t* results)
{
    uint8_t* d_pages = nullptr;
    uint64_t* d_results = nullptr;
    
    // 分配设备内存
    cudaMalloc(&d_pages, num_pages * page_size);
    cudaMalloc(&d_results, num_pages * sizeof(uint64_t));
    
    // 拷贝数据到设备 (优化：异步拷贝)
    for (size_t i = 0; i < num_pages; ++i) {
        cudaMemcpy(
            d_pages + i * page_size,
            pages[i],
            page_size,
            cudaMemcpyHostToDevice
        );
    }
    
    // 计算线程块大小 (每页256个线程)
    const int threads_per_page = 256; 
    dim3 blocks(num_pages);
    dim3 threads(threads_per_page);

    gpu_hash_kernel_batch<<<blocks, threads>>>(
        d_pages, page_size, num_pages, d_results
    );
    
    cudaMemcpy(
        results,
        d_results,
        num_pages * sizeof(uint64_t),
        cudaMemcpyDeviceToHost
    );

    cudaFree(d_pages);
    cudaFree(d_results);
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