#include <stdio.h>
#include <time.h>
#include <string>
#include <cstring>
#include <openssl/sha.h> // 用于计算SHA256哈希
#include <xxh3.h>
#include <cuda_runtime.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <array>
#include <iomanip>
#include <sstream>

#define PRIME64_1 0x9E3779B185EBCA87ULL
#define PRIME64_2 0xC2B2AE3D27D4EB4FULL
#define PRIME64_3 0x165667B19E3779F9ULL
#define PRIME64_4 0x85EBCA77C2B2AE63ULL
#define PRIME64_5 0x27D4EB2F165667C5ULL

enum ApiId {
    API_CU_MEM_ALLOC_HOOK,
    API_CU_MEM_ALLOC_ORIG,
    API_CU_MEM_ADDRESS_RESERVE_ORIG,
    API_CU_MEMCPY_HTOD_ASYNC_HOOK,
    API_CU_MEMCPY_HTOD_ASYNC_ORIG,
    API_CU_MEM_CREATE_HOOK,
    API_CU_MEM_CREATE_ORIG,
    API_CU_MEM_MAP_HOOK,
    API_CU_MEM_MAP_ORIG,
    API_CU_MEM_UNMAP_HOOK,
    API_CU_MEM_UNMAP_ORIG,
    API_CU_MEM_SETACCESS_HOOK,
    API_CU_MEM_SETACCESS_ORIG,
    API_CU_MEMCPY_DTO_D_HOOK,
    API_CU_MEMCPY_DTO_D_ORIG,
    API_CU_MEM_RELEASE_ORIG,
    API_CU_MEM_EXPORT_ORIG,
    API_CU_MEM_IMPORT_ORIG,
    API_TRY_DEDUPLICATE_PAGE,
    API_USERFAULT_HANDLE,
    API_HASH_SHA256,
    API_HASH_SHA256_GPU,
    API_HASH_XXH64,
    API_HASH_XXH64_GPU,
    API_COUNT
};

struct ApiStat {
    std::atomic<uint64_t> calls{0};
    std::atomic<uint64_t> total_ns{0};
};

static std::array<ApiStat, API_COUNT> g_api_stats;

static inline uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct ScopedTimer {
    int id;
    uint64_t start;
    ScopedTimer(int id_) : id(id_), start(now_ns()) {}
    ~ScopedTimer() {
        uint64_t dur = now_ns() - start;
        g_api_stats[id].calls.fetch_add(1, std::memory_order_relaxed);
        g_api_stats[id].total_ns.fetch_add(dur, std::memory_order_relaxed);
    }
};

static const char* g_api_names[API_COUNT] = {
    "cuMemAlloc_hook",
    "cuMemAlloc_orig",
    "cuMemAddressReserve_orig",
    "cuMemcpyHtoDAsync_hook",
    "cuMemcpyHtoDAsync_orig",
    "cuMemCreate_hook",
    "cuMemCreate_orig",
    "cuMemMap_hook",
    "cuMemMap_orig",
    "cuMemUnmap_hook",
    "cuMemUnmap_orig",
    "cuMemSetAccess_hook",
    "cuMemSetAccess_orig",
    "cuMemcpyDtoD_hook",
    "cuMemcpyDtoD_orig",
    "cuMemRelease_orig",
    "cuMemExport_orig",
    "cuMemImport_orig",
    "try_deduplicate_page",
    "userfault_handler_thread",
    "hash_sha256",
    "hash_sha256_gpu",
    "hash_xxh64",
    "hash_xxh64_gpu"
};


std::string current_time();
std::string format_bytes(size_t bytes);
std::string sha256(const void* data, size_t size);
std::string fast_hash(const void* data, size_t size);
std::string gpu_fast_hash_single(unsigned long long d_data, size_t size);
void get_command_line();