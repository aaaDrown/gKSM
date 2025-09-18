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

////////////////////////////////////////////////
// XXH64
////////////////////////////////////////////////
std::string fast_hash(const void* data, size_t size) {
    XXH64_hash_t hash = XXH3_64bits(data, size);
    std::ostringstream oss;
    oss << std::hex << std::setfill('0') << std::setw(16) << hash;
    return oss.str();
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