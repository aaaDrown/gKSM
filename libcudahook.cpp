#include "libcudahook.h"
#include <sys/ioctl.h>
#include <cstdlib>

static CUresult(* orig_cuMemAlloc)(CUdeviceptr * dptr, size_t bytesize) = NULL;
static CUresult(* orig_cuMemAlloc_v2)(CUdeviceptr * dptr, size_t bytesize) = NULL;
static CUresult CUDAAPI(*orig_cuMemcpyHtoD)(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount) = NULL;
static CUresult CUDAAPI(*orig_cuMemcpyDtoD)(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) = NULL;
static CUresult CUDAAPI(*orig_cuMemcpyDtoH)(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount) = NULL;
static CUresult CUDAAPI(*orig_cuMemcpyHtoDAsync)(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream) = NULL;
static CUresult CUDAAPI(*orig_cuMemcpyHtoD_v2)(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount) = NULL;
static CUresult( * orig_cuCtxCreate)(CUcontext * pctx, unsigned int flags, CUdevice dev) = NULL;
static CUresult( * orig_cuCtxCreate_v2)(CUcontext * pctx, unsigned int flags, CUdevice dev) = NULL;
static CUresult( * orig_cuCtxCreate_v3)(CUcontext * pctx, CUexecAffinityParam * paramsArray, int numParams, unsigned int flags, CUdevice dev) = NULL;
static CUresult( * orig_cuMemFree)(CUdeviceptr dptr) = NULL;

#define DECLARE_ORIG_FUNC(name) static decltype(name)* orig_##name = nullptr
DECLARE_ORIG_FUNC(cuGetProcAddress);
DECLARE_ORIG_FUNC(cuGetProcAddress_v2);
// DECLARE_ORIG_FUNC(cuInit);
// cudaMalloc  cudaMemcpy
// VMM API函数声明
DECLARE_ORIG_FUNC(cuMemCreate); 
DECLARE_ORIG_FUNC(cuMemAddressReserve);
DECLARE_ORIG_FUNC(cuMemAddressFree);
DECLARE_ORIG_FUNC(cuMemMap);
DECLARE_ORIG_FUNC(cuMemSetAccess);
DECLARE_ORIG_FUNC(cuMemUnmap);
DECLARE_ORIG_FUNC(cuMemRelease);
DECLARE_ORIG_FUNC(cuMemGetAllocationGranularity);
DECLARE_ORIG_FUNC(cuMemExportToShareableHandle);
DECLARE_ORIG_FUNC(cuMemImportFromShareableHandle);
DECLARE_ORIG_FUNC(cuMemGetAccess);
DECLARE_ORIG_FUNC(cuCtxGetCurrent);

static char lib_name[20] = "";
static std::mutex g_device_ctx_mutex;
static std::map<int, CUcontext> g_device_contexts;
static std::map<int, bool> g_userfault_registered_devices;
static std::map<int, bool> g_missing_ctx_warnings;

int get_current_device_id() {
    CUdevice cu_device = -1;
    CUresult res = cuCtxGetDevice(&cu_device);
    if (res != CUDA_SUCCESS) {
        return -1;
    }
    return static_cast<int>(cu_device);
}

int get_current_dedup_domain_id() {
    const char* env = getenv("CUDA_DEDUP_DEVICE_ID");
    if (env && *env) {
        char* end = nullptr;
        long val = strtol(env, &end, 10);
        if (end != env) {
            return static_cast<int>(val);
        }
    }
    return get_current_device_id();
}

void register_current_ctx_for_device() {
    if (!orig_cuCtxGetCurrent) {
        return;
    }

    CUcontext ctx = nullptr;
    if (orig_cuCtxGetCurrent(&ctx) != CUDA_SUCCESS || ctx == nullptr) {
        return;
    }

    int device_id = get_current_device_id();
    if (device_id < 0) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_device_ctx_mutex);
    g_device_contexts[device_id] = ctx;
}

bool switch_to_device_ctx(int device_id) {
    if (device_id < 0) {
        return false;
    }

    CUcontext ctx = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_device_ctx_mutex);
        auto it = g_device_contexts.find(device_id);
        if (it != g_device_contexts.end()) {
            ctx = it->second;
        } else if (!g_missing_ctx_warnings[device_id]) {
            g_missing_ctx_warnings[device_id] = true;
            fprintf(stderr, "Missing CUDA context for device %d\n", device_id);
        }
    }

    if (ctx == nullptr) {
        return false;
    }

    CUresult res = cuCtxSetCurrent(ctx);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "cuCtxSetCurrent failed for device %d: %d\n", device_id, res);
        return false;
    }
    return true;
}

bool ensure_userfault_registration_for_current_device(int device_id) {
    if (userfault_fd == -1 || device_id < 0) {
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(g_device_ctx_mutex);
        auto it = g_userfault_registered_devices.find(device_id);
        if (it != g_userfault_registered_devices.end() && it->second) {
            return true;
        }
    }

    if (!switch_to_device_ctx(device_id)) {
        return false;
    }

    CUdevice cu_device = -1;
    if (cuCtxGetDevice(&cu_device) != CUDA_SUCCESS) {
        fprintf(stderr, "cuCtxGetDevice failed while registering userfault for device %d\n", device_id);
        return false;
    }

    CUuuid uuid;
    if (cuDeviceGetUuid(&uuid, cu_device) != CUDA_SUCCESS) {
        fprintf(stderr, "cuDeviceGetUuid failed for device %d\n", device_id);
        return false;
    }

    int ret = ioctl(userfault_fd, CMD_START_USERFAULT, *(unsigned long*)&uuid);
    if (ret == -1) {
        fprintf(stderr, "ioctl CMD_START_USERFAULT failed for device %d\n", device_id);
        return false;
    }

    std::lock_guard<std::mutex> lock(g_device_ctx_mutex);
    g_userfault_registered_devices[device_id] = true;
    return true;
}

__attribute__((constructor))
static void init_hook() {
    Dl_info info;
    static int count = 0;
    if (dladdr((void*)init_hook, &info)) {
        printf("[%d] INIT HOOK called from %s (base: %p)\n", 
               ++count, info.dli_fname, info.dli_fbase);
        strcpy(lib_name,info.dli_fname);
    } else {
        printf("[%d] INIT HOOK called (unknown origin)\n", ++count);
    }

    // libcuda.so和libcuda.so.1只加载一个就行了
    if(!strcmp(lib_name,"./build/libcuda.so")) return;

    const char* config_file_path = "/root/dedup/hook.conf";

    parse_config_file(config_file_path, g_hook_config);
    // print_config(g_hook_config);

    g_page_size = g_hook_config.page_size;
    g_page_map = new std::map<CUdeviceptr, LocalPageInfo>();
    g_dedup_pid = getpid();
    
    printf("Initializing CUDA Driver API Hook...\n");
    init_hook_api();
    printf("CUDA Driver API Hook initialized, Page Size: %s, pid: %d, ppid:%d\n", format_bytes(g_page_size).c_str(), g_dedup_pid, getppid());
    
    printf("Initializing ipc...\n");
    init_ipc();
    printf("ipc initialized\n");
}

SharedPageInfo* find_page_by_hash(const char* hash, int dedup_domain_id) {
    int i;
    for (i = 0; i < MAX_DEDUPE_PAGES; i++) {
        if (g_shared_memory->shared_pages[i].valid &&
            g_shared_memory->shared_pages[i].dedup_domain_id == dedup_domain_id &&
            strncmp(g_shared_memory->shared_pages[i].hash, hash, HASH_SIZE) == 0) {
            return &g_shared_memory->shared_pages[i];
        }
    }
    return nullptr;
}


void* background_scan_thread(void* arg) {
    CUcontext cuda_ctx = (CUcontext)arg;
    if (cuda_ctx != nullptr) {
        CUresult res = cuCtxSetCurrent(cuda_ctx);
        if (res != CUDA_SUCCESS) {
            fprintf(stderr, "cuCtxSetCurrent failed: %d\n", res);
        }
        register_current_ctx_for_device();
    }

    const int initial_scan_interval_ms = g_hook_config.initial_scan_interval_ms; 
    const int max_scan_interval_ms = g_hook_config.max_scan_interval_ms;    
    const int stable_threshold = g_hook_config.stable_threshold;            
    
    int scan_interval_ms = initial_scan_interval_ms;
    int consecutive_no_dedup = 0;

    usleep(g_hook_config.boot_scan_interval_ms * 1000);
    while (g_background_scanning) {
        usleep(scan_interval_ms * 1000);
        //ScopedTimer t_hook(API_BACKGROUND_SCAN);
        g_background_scans++;
        
        size_t pages_scanned = 0;
        pages_deduped = 0;
        size_t hash_calculated = 0;
        size_t stable_candidates = 0;
        size_t bloom_filtered = 0;  // 被布隆过滤器过滤掉的页面数
        
        uint64_t start = now_ns();
        std::vector<CUdeviceptr> to_scan;
        {
            const int max_scan_pages = g_hook_config.max_scan_pages;
            for (auto& page_item : *g_page_map) {
                if (page_item.second.is_deduped == false) {
                    to_scan.push_back(page_item.first);
                    if (max_scan_pages > 0 && to_scan.size() >= max_scan_pages) {
                        break;
                    }
                }
            }
        }
        
        pages_scanned = to_scan.size();
        printf("[%s] Starting background deduplication scan #%zu scan size=%zu (interval: %dms)\n", 
               current_time().c_str(), g_background_scans, pages_scanned, scan_interval_ms);
        
        const size_t max_batch_size = g_hook_config.max_batch_size;
        size_t cur = 0;
        
        while (cur < to_scan.size()) {
            std::vector<CUdeviceptr> batch_pages;
            batch_pages.reserve(max_batch_size);
            int batch_device_id = -1;

            while (cur < to_scan.size() && batch_pages.size() < max_batch_size) {
                auto page_it = g_page_map->find(to_scan[cur]);
                if (page_it == g_page_map->end()) {
                    cur++;
                    continue;
                }

                const int page_device_id = page_it->second.device_id;
                if (batch_device_id == -1) {
                    batch_device_id = page_device_id;
                } else if (page_device_id != batch_device_id) {
                    break;
                }

                batch_pages.push_back(page_it->first);
                cur++;
            }

            if (batch_pages.empty()) {
                continue;
            }

            if (!switch_to_device_ctx(batch_device_id)) {
                continue;
            }

            std::vector<std::string> batch_results;
            if (batch_pages.size() > 1) {
                std::vector<unsigned long long> page_ptrs;
                page_ptrs.reserve(batch_pages.size());
                for (CUdeviceptr va : batch_pages) {
                    page_ptrs.push_back(static_cast<unsigned long long>(va));
                }

                {
                    ScopedTimer t_hook(API_HASH_XXH64_GPU_BATCH);
                    batch_results = gpu_fast_hash_batch(page_ptrs.data(), g_page_size, batch_pages.size());
                }
            } else {
                batch_results.push_back(gpu_fast_hash_single(batch_pages[0], g_page_size));
            }

            for (size_t j = 0; j < batch_pages.size(); j++) {
                CUdeviceptr va = batch_pages[j];
                auto local_it = g_page_map->find(va);
                if (local_it == g_page_map->end()) {
                    continue;
                }

                LocalPageInfo& page_info = local_it->second;
                const char* current_hash = batch_results[j].c_str();
                hash_calculated++;

                if (current_hash[0] == '\0') continue;

                {
                    bool all_zero = true;
                    for (int k = 0; k < HASH_SIZE - 1 && current_hash[k] != '\0'; ++k) {
                        if (current_hash[k] != '0') { all_zero = false; break; }
                    }
                    if (all_zero) continue;
                }

                if (page_info.hash[0] == '\0') {
                    strncpy(page_info.prev_hash, current_hash, HASH_SIZE);
                } else {
                    strncpy(page_info.prev_hash, page_info.hash, HASH_SIZE);
                }
                strncpy(page_info.hash, current_hash, HASH_SIZE);

                if (strncmp(page_info.prev_hash, page_info.hash, HASH_SIZE) == 0) {
                    page_info.unchanged_scans++;

                    if (page_info.unchanged_scans >= stable_threshold) {
                        page_info.is_stable = true;
                        stable_candidates++;

                        SharedPageInfo* shared_page = find_page_by_hash(current_hash, page_info.dedup_domain_id);
                        if (shared_page) {
                            if (try_deduplicate_page(va, NULL, g_page_size, current_hash)) {
                                page_info.is_deduped = true;
                            } else {
                                fprintf(stderr, "try_deduplicate_page failed in scan mode\n");
                            }
                        } else {
                            bool found_candidate = false;
                            for (int i = 0; i < MAX_DEDUPE_PAGES; i++) {
                                CandidatePageInfo& candidate = g_shared_memory->candidate_pages[i];
                                if (!candidate.valid ||
                                    candidate.dedup_domain_id != page_info.dedup_domain_id ||
                                    strncmp(candidate.hash, current_hash, HASH_SIZE) != 0) {
                                    continue;
                                }

                                if (candidate.pid == g_dedup_pid && candidate.va == va) {
                                    continue;
                                }

                                found_candidate = true;
                                if (candidate.pid == g_dedup_pid) {
                                    if (try_deduplicate_page(va, NULL, g_page_size, current_hash) &&
                                        try_deduplicate_page(candidate.va, NULL, g_page_size, current_hash)) {
                                        auto it = g_page_map->find(va);
                                        if (it != g_page_map->end()) {
                                            it->second.is_deduped = true;
                                        }
                                        it = g_page_map->find(candidate.va);
                                        if (it != g_page_map->end()) {
                                            it->second.is_deduped = true;
                                        }
                                        candidate.valid = false;
                                        g_shared_memory->candidate_page_count--;
                                    }
                                } else {
                                    bool current_page_deduped = try_deduplicate_page(va, NULL, g_page_size, current_hash);
                                    if (current_page_deduped) {
                                        auto it = g_page_map->find(va);
                                        if (it != g_page_map->end()) {
                                            it->second.is_deduped = true;
                                        }
                                        pid_t target_pid = candidate.pid;
                                        CUdeviceptr target_va = candidate.va;
                                        unsigned long long target = (unsigned long long)target_va;
                                        if (ipc_send_dedup_request(target_pid, target, current_hash, candidate.device_id, page_info.dedup_domain_id)) {
                                            candidate.valid = false;
                                            g_shared_memory->candidate_page_count--;
                                        }
                                    }
                                }

                                break;
                            }

                            if (!found_candidate) {
                                for (int i = 0; i < MAX_DEDUPE_PAGES; i++) {
                                    CandidatePageInfo& candidate = g_shared_memory->candidate_pages[i];
                                    if (!candidate.valid) {
                                        candidate.valid = true;
                                        strncpy(candidate.hash, current_hash, HASH_SIZE);
                                        candidate.pid = g_dedup_pid;
                                        candidate.va = va;
                                        candidate.device_id = page_info.device_id;
                                        candidate.dedup_domain_id = page_info.dedup_domain_id;
                                        candidate.unchanged_scans = page_info.unchanged_scans;
                                        g_shared_memory->candidate_page_count++;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                } else {
                    page_info.unchanged_scans = 0;
                    page_info.is_stable = false;

                    for (int i = 0; i < MAX_DEDUPE_PAGES; i++) {
                        CandidatePageInfo& candidate = g_shared_memory->candidate_pages[i];
                        if (candidate.valid &&
                            candidate.dedup_domain_id == page_info.dedup_domain_id &&
                            candidate.va == va &&
                            candidate.pid == g_dedup_pid) {
                            candidate.valid = false;
                            g_shared_memory->candidate_page_count--;
                            break;
                        }
                    }
                }
            }
        }

        if (pages_deduped == 0) {
            consecutive_no_dedup++;
            scan_interval_ms = std::min(max_scan_interval_ms, scan_interval_ms * 2);
        } else {
            consecutive_no_dedup = 0;
            scan_interval_ms = initial_scan_interval_ms;
        }
        uint64_t dur = now_ns() - start;
        double total_ms = double(dur) / 1e6;
        printf("[%s] Background scan completed. Page total: %d, Scanned: %zu, Bloom filtered: %zu, Hash calculated: %zu, "
               "Stable candidates: %zu, Deduped: %zu, New interval: %dms, total_ms: %.3f\n",
               current_time().c_str(), 
               g_page_map->size(),
               pages_scanned, 
               bloom_filtered,
               hash_calculated,
               stable_candidates,
               pages_deduped,
               scan_interval_ms,
               total_ms);
        
        printf("\n==== CUDA-HOOK API TIMING SUMMARY ====\n");
        printf("%-32s %12s %12s %12s\n", "API", "calls", "total(ms)", "avg(us)");
        for (int i = 0; i < API_COUNT; ++i) {
            // 使用acquire内存序确保看到最新的更新
            uint64_t calls = g_api_stats[i].calls.load(std::memory_order_acquire);
            uint64_t total_ns = g_api_stats[i].total_ns.load(std::memory_order_acquire);
            if (calls == 0) continue;
            double total_ms = double(total_ns) / 1e6;
            double avg_us = double(total_ns) / double(calls) / 1e3;
            printf("%-32s %12llu %12.3f %12.3f\n",
                   g_api_names[i],
                   (unsigned long long)calls,
                   total_ms,
                   avg_us);
        }
        printf("======================================\n");
    }
    
    return NULL;
}

void* ipc_handler_thread(void* arg) {
    CUcontext cuda_ctx = (CUcontext)arg;
    if (cuda_ctx != nullptr) {
        CUresult res = cuCtxSetCurrent(cuda_ctx);
        if (res == CUDA_SUCCESS) {
            register_current_ctx_for_device();
        }
    }
    while (g_ipc_state.running) {
        int client = accept(g_ipc_state.ipc_sock, NULL, NULL);
        if (client < 0) {
            if (g_ipc_state.running) perror("accept failed");
            continue;
        }
        
        char request_type;
        if (read(client, &request_type, 1) != 1) {
            close(client);
            continue;
        }
        
        // 处理不同类型的请求
        switch (request_type) {
            case 'R': { // 请求共享页面文件描述符
                SharedFdRequest req = {};
                if (read(client, &req, sizeof(req)) != sizeof(req)) {
                    close(client);
                    continue;
                }

                if (!switch_to_device_ctx(req.device_id)) {
                    close(client);
                    continue;
                }
                
                CUmemGenericAllocationHandle handle = 0;
                {
                    SharedPageInfo* page = find_page_by_hash(req.hash, req.dedup_domain_id);
                    if (page && page->owner_pid == g_dedup_pid) {
                        auto page_it = g_page_map->find(page->owner_va);
                        if (page_it != g_page_map->end()) {
                            handle = page_it->second.handle;
                        }
                    }
                }
                
                if (handle) {
                    int shareable_fd;
                    CUresult res = orig_cuMemExportToShareableHandle(
                        (void*)&shareable_fd,
                        handle,
                        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                        0
                    );
                    
                    if (res == CUDA_SUCCESS) {
                        ipc_send_fd(client, shareable_fd);
                        close(shareable_fd); 
                    } else {
                        fprintf(stderr, "Export failed in IPC handler: %d\n", res);
                    }
                } else {
                    printf("Hash not found in local pages: %s device=%d domain=%d\n", req.hash, req.device_id, req.dedup_domain_id);
                }
                break;
            }
            
            case 'T': { // 所有权转移请求
                struct IpcRequest req;
                if (read(client, &req, sizeof(req)) != sizeof(req)) {
                    close(client);
                    continue;
                }

                IpcResponse resp = {0};
                SharedPageInfo* page;
                if (!switch_to_device_ctx(req.device_id)) {
                    write(client, &resp, sizeof(resp));
                    break;
                }

                page = find_page_by_hash(req.hash, req.dedup_domain_id);
                if (page && page->owner_pid == req.src_pid) {
                    auto page_it = g_page_map->find(page->owner_va);
                    if (page_it != g_page_map->end()) {
                        LocalPageInfo& page = page_it->second;
                        resp.success = 1;
                        printf("Accepted ownership for page (Hash:%s)\n", req.hash);
                    }
                }
                write(client, &resp, sizeof(resp));
                break;
            }
            
            case DEDUP_REQUEST_MSG: { // 处理重删请求
                DedupRequest req;
                if (read(client, &req, sizeof(req)) != sizeof(req)) {
                    close(client);
                    continue;
                }
                // printf("req.va=%lx req.hash=%s\n", req.va, req.hash);

                bool success = false;
                auto req_page_it = g_page_map->find((CUdeviceptr)req.va);
                if (req_page_it != g_page_map->end() &&
                    req_page_it->second.dedup_domain_id == req.dedup_domain_id &&
                    switch_to_device_ctx(req.device_id)) {
                    success = try_deduplicate_page((CUdeviceptr)req.va, NULL, g_page_size, req.hash);
                }
                
                if (success) {
                    //std::lock_guard<std::mutex> lock(g_mutex);
                    auto it = g_page_map->find((CUdeviceptr)req.va);
                    if (it != g_page_map->end()) {
                        it->second.is_deduped = true;
                    }
                }

                DedupResponse response;
                response.type = DEDUP_RESPONSE_MSG;
                response.success = success;
                write(client, &response, sizeof(response));
                break;
            }
            
            default:
                printf("Unknown request type: %c\n", request_type);
                break;
        }
        
        close(client);
    }
    return NULL;
}

void init_ipc() {
    CUcontext ctx = nullptr;
    orig_cuCtxGetCurrent(&ctx);
    g_ipc_state.ipc_sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (g_ipc_state.ipc_sock < 0) {
        perror("socket creation failed");
        return;
    }

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    std::string sock_path = ipc_get_current_sock_path();
    strncpy(addr.sun_path, sock_path.c_str(), sizeof(addr.sun_path)-1);
    
    unlink(addr.sun_path);
    
    if (bind(g_ipc_state.ipc_sock, (struct sockaddr*)&addr, sizeof(addr))) {
        perror("bind failed");
        close(g_ipc_state.ipc_sock);
        g_ipc_state.ipc_sock = -1;
        return;
    }
    
    if (listen(g_ipc_state.ipc_sock, 5)) {
        perror("listen failed");
        close(g_ipc_state.ipc_sock);
        g_ipc_state.ipc_sock = -1;
        return;
    }

    g_ipc_state.running = true;
    if (pthread_create(&g_ipc_state.handler_thread, NULL, ipc_handler_thread, (void *)ctx)) {
        perror("pthread_create failed");
        g_ipc_state.running = false;
        return;
    }
}

void* userfault_handler_thread(void* arg) {
    int fd = userfault_fd, ret;
    unsigned long long fault_addr;
    pid_t cur_pid;
    struct fault_info info;
    CUcontext cuda_ctx = (CUcontext)arg;
    size_t page_mask = ~(g_page_size - 1);

    CUresult res = CUDA_SUCCESS;
    if (cuda_ctx != nullptr) {
        res = cuCtxSetCurrent(cuda_ctx);
        if (res != CUDA_SUCCESS) {
            fprintf(stderr, "cuCtxSetCurrent failed: %d\n", res);
        } else {
            register_current_ctx_for_device();
        }
    }
    
    while(g_userfault_handling) {
        info.fault_addr = 0;
        ret = read(fd, (void *)&info, 16);
        if (ret == -1) {
            printf("read error, errno:%d\n", errno);
            break;
        }
        if (ret != 16) continue;
        fault_addr = info.fault_addr;
        cur_pid = info.pid;
        CUdeviceptr base_addr = (CUdeviceptr)((uintptr_t)fault_addr & page_mask);
        SharedPageInfo* shared_page = nullptr;
        auto it = g_page_map->find(base_addr);
        if (it == g_page_map->end()) {
            fprintf(stdout, "[%s] get fault_addr:0x%lx  base_addr:0x%lx  cur_pid:%d  fault_type:%d\n", current_time().c_str(), fault_addr, base_addr, cur_pid, info.fault_type);
            auto lb = g_page_map->lower_bound(base_addr);
            if (lb != g_page_map->end()) {
                fprintf(stdout, "no page entry for base_addr=%lx, next entry=%lx\n", base_addr, lb->first);
            } else if (!g_page_map->empty()) {
                auto last = std::prev(g_page_map->end());
                fprintf(stdout, "no page entry for base_addr=%lx, last entry=%lx\n", base_addr, last->first);
            } else {
                fprintf(stdout, "g_page_map empty\n");
            }

            write(fd, "done", 4);
            continue;
        }
        LocalPageInfo& fault_page = it->second;
        if (!switch_to_device_ctx(fault_page.device_id)) {
            write(fd, "done", 4);
            continue;
        }
        switch (info.fault_type) {
            case 0: { //处理延迟分配
                CUmemAllocationProp prop;
                memset(&prop, 0, sizeof(prop));
                prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
                prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                prop.location.id = fault_page.device_id;
                prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
                CUmemGenericAllocationHandle handle;

                {
                    //ScopedTimer t_hook_create(API_CU_MEM_CREATE_HOOK);
                    res = cuMemCreate(&handle, g_page_size, &prop, 0);
                }
                if (res != CUDA_SUCCESS) {
                    fprintf(stderr, "cuMemCreate failed: %d\n", res);
                    break;
                }
        
                {
                    //ScopedTimer t_orig_map(API_CU_MEM_MAP_ORIG);
                    res = orig_cuMemMap(base_addr, g_page_size, 0, handle, 0);
                }
                if (res != CUDA_SUCCESS) {
                    fprintf(stderr, "cuMemMap failed: %d\n", res);
                    orig_cuMemRelease(handle);
                    break;
                }

                CUmemAccessDesc accessDesc = {};
                accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                accessDesc.location.id = fault_page.device_id;
                accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                
                {
                    //ScopedTimer t_orig_setaccess(API_CU_MEM_SETACCESS_ORIG);
                    res = orig_cuMemSetAccess(base_addr, g_page_size, &accessDesc, 1);
                }
                if (res != CUDA_SUCCESS) {
                    fprintf(stderr, "cuMemSetAccess failed: %d\n", res);
                    orig_cuMemUnmap(base_addr, g_page_size);
                    break;
                }

                fault_page.handle = handle;
                
                write(fd, "done", 4);
                break;
            }
            case 3: { //处理写时复制
                // printf("[%s] get fault_addr:0x%lx  base_addr:0x%lx  cur_pid:%d  fault_type:%d  fault_page.hash:%s\n", current_time().c_str(), fault_addr, base_addr, cur_pid, info.fault_type, fault_page.hash);
                {
                    // std::lock_guard<std::mutex> lock(g_mutex);
                    if (fault_page.pagefault_flag == true) {
                        ret = write(fd, "done", 4);
                        if (ret == -1) {
                            printf("write error, errno:%d\n", errno);
                            break;
                        }
                        continue;
                    }
                    fault_page.pagefault_flag = true;
                }

                //ScopedTimer t_hook(API_USERFAULT_HANDLE);
                // pthread_mutex_lock(&g_shared_memory->mutex);
                {
                    //ScopedTimer t_hook(API_FIND_PAGE_BY_HASH); 
                    shared_page = find_page_by_hash(fault_page.hash, fault_page.dedup_domain_id);
                }

                if (shared_page != nullptr) {
                    if (shared_page->ref_count > 0)
                        shared_page->ref_count--;
                    if (g_dedup_pid == shared_page->owner_pid && base_addr == shared_page->owner_va) { // owner页
                        if (shared_page->ref_count == 0) { 
                            shared_page->valid = false;
                            g_shared_memory->shared_page_count--;

                            CUmemAccessDesc accessDesc = {
                                {CU_MEM_LOCATION_TYPE_DEVICE, fault_page.device_id},
                                CU_MEM_ACCESS_FLAGS_PROT_READWRITE
                            };
                            if (orig_cuMemSetAccess(base_addr, g_page_size, &accessDesc, 1) != CUDA_SUCCESS) {
                                printf("Failed to set access for new page\n");
                                break;
                            }
                        } else {
                            int new_owner_pid = 0;
                            CUdeviceptr new_owner_va = 0;
                            bool found_new_owner = false;
                            for (int i = 0; i < MAX_SHARED_REFS; i++) { // 优先寻找进程内部的
                                if (shared_page->ref_pids[i] != 0 && shared_page->ref_pids[i] == g_dedup_pid) {
                                    new_owner_pid = shared_page->ref_pids[i];
                                    new_owner_va = shared_page->ref_vas[i];
                                    found_new_owner = true;
                                    shared_page->ref_pids[i] = 0;
                                    shared_page->ref_vas[i] = 0;
                                    break;
                                }
                            }

                            if (!found_new_owner) {
                                for (int i = 0; i < MAX_SHARED_REFS; i++) { 
                                    if (shared_page->ref_pids[i] != 0) {
                                        new_owner_pid = shared_page->ref_pids[i];
                                        new_owner_va = shared_page->ref_vas[i];
                                        shared_page->ref_pids[i] = 0;
                                        shared_page->ref_vas[i] = 0;
                                        found_new_owner = true;
                                        break;
                                    }
                                }
                            }
                            // printf("Selected new owner: PID=%d, VA=0x%lx\n", new_owner_pid, new_owner_va);
                            
                            if (!found_new_owner) {
                                printf("No other references found, releasing shared page\n");
                                shared_page->valid = false;
                                g_shared_memory->shared_page_count--;
                                return NULL;
                            }

                            if (new_owner_pid == g_dedup_pid) { // 转移物理handle的所有权就可以了
                                LocalPageInfo& new_owner_page = (g_page_map->find(new_owner_va))->second;
                                CUmemGenericAllocationHandle new_handle;
                                CUmemAllocationProp prop;
                                memset(&prop, 0, sizeof(prop));
                                prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
                                prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                                prop.location.id = fault_page.device_id;
                                prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

                                if (orig_cuMemCreate(&new_handle, g_page_size, &prop, 0) == CUDA_SUCCESS) {
                                    if(orig_cuMemUnmap(base_addr, g_page_size) != CUDA_SUCCESS) {
                                        printf("userfault handle in orig_cuMemUnmap error\n");
                                        break;
                                    }

                                    if(orig_cuMemMap(base_addr, g_page_size, 0, new_handle, 0) != CUDA_SUCCESS) {
                                        printf("userfault handle in orig_cuMemMap error\n");
                                        break;
                                    }

                                    CUmemAccessDesc accessDesc = {
                                        {CU_MEM_LOCATION_TYPE_DEVICE, fault_page.device_id},
                                        CU_MEM_ACCESS_FLAGS_PROT_READWRITE
                                    };
                                    orig_cuMemSetAccess(base_addr, g_page_size, &accessDesc, 1);
                                    // TODO:这里如果利用页内数据对页内进行写入需要先复制再写
                                    // CUresult res = orig_cuMemcpyDtoD(base_addr, new_owner_va, g_page_size);
                                    // if(res != CUDA_SUCCESS){
                                    //     printf("switch owner error in orig_cuMemcpyDtoD : %d\n", res);
                                    // } 
                                }

                                shared_page->owner_pid = g_dedup_pid;
                                shared_page->owner_va = new_owner_va;
                                new_owner_page.handle = fault_page.handle;
                                fault_page.handle = new_handle;
                                fault_page.is_exported = false;
                                g_saved_memory -= g_page_size;
                            } else { 
                                if (new_owner_pid != 0) {                        
                                    // printf("Ownership transferred to PID=%d\n", new_owner_pid); 
                                    
                                    CUmemGenericAllocationHandle new_handle;
                                    CUmemAllocationProp prop;
                                    memset(&prop, 0, sizeof(prop));
                                    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
                                    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                                    prop.location.id = fault_page.device_id;
                                    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

                                    if (orig_cuMemCreate(&new_handle, g_page_size, &prop, 0) == CUDA_SUCCESS) {
                                        if(orig_cuMemUnmap(base_addr, g_page_size) != CUDA_SUCCESS) {
                                            printf("userfault handle in orig_cuMemUnmap error\n");
                                            break;
                                        }

                                        if(orig_cuMemMap(base_addr, g_page_size, 0, new_handle, 0) != CUDA_SUCCESS) {
                                            printf("userfault handle in orig_cuMemMap error\n");
                                            break;
                                        }

                                        CUmemAccessDesc accessDesc = {
                                            {CU_MEM_LOCATION_TYPE_DEVICE, fault_page.device_id},
                                            CU_MEM_ACCESS_FLAGS_PROT_READWRITE
                                        };
                                        orig_cuMemSetAccess(base_addr, g_page_size, &accessDesc, 1);
                                    }
                                    fault_page.handle = new_handle;
                                    fault_page.is_exported = false;
                                    shared_page->owner_pid = new_owner_pid;
                                    shared_page->owner_va = new_owner_va;
                                    g_saved_memory -= g_page_size;
                                }
                            }                    
                        }
                    } else {// 非owner页重删
                        if (shared_page->ref_count <= 0) {
                            printf("userfault handle fail g_dedup_pid=%d  shared_page->owner_pid=%d  shared_page->ref_count=%d\n", g_dedup_pid, shared_page->owner_pid, shared_page->ref_count);
                            ret = write(fd, "done", 4);
                            break;
                        }
                        CUmemAllocationProp prop = {};
                        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
                        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                        prop.location.id = fault_page.device_id;
                        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
                        
                        CUmemGenericAllocationHandle new_handle;
                        if (orig_cuMemCreate(&new_handle, g_page_size, &prop, 0) != CUDA_SUCCESS) {
                            printf("Failed to allocate new page for CoW\n");
                            continue;
                        }
                        
                        if (orig_cuMemUnmap(base_addr, g_page_size) != CUDA_SUCCESS) {
                            printf("Failed to unmap original VA\n");
                            orig_cuMemRelease(new_handle);
                            continue;
                        } 

                        if (orig_cuMemMap(base_addr, g_page_size, 0, new_handle, 0) != CUDA_SUCCESS) {
                            printf("Failed to map new handle to original VA\n");
                            orig_cuMemRelease(new_handle);
                            continue;
                        } 

                        CUmemAccessDesc accessDesc;
                        memset(&accessDesc, 0, sizeof(accessDesc));
                        accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                        accessDesc.location.id = fault_page.device_id;
                        accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

                        res = orig_cuMemSetAccess(base_addr, g_page_size, &accessDesc, 1);
                        if (res != CUDA_SUCCESS) {
                            printf("Failed to set access for new page: res=%d\n", res);
                            orig_cuMemUnmap(base_addr, g_page_size);
                            orig_cuMemMap(base_addr, g_page_size, 0, fault_page.handle, 0);
                            orig_cuMemRelease(new_handle);
                            continue;
                        }

                        // TODO 这里的fault address需要重新复制
                        // CUresult res = orig_cuMemcpyDtoD(base_addr, shared_page->owner_va, g_page_size);
                        // if(res != CUDA_SUCCESS){
                        //     printf("Failed in orig_cuMemcpyDtoD : %d\n", res);
                        // } 
                        
                        int i;
                        for(i = 0;i < MAX_SHARED_REFS;i++){
                            if(shared_page->ref_vas[i] == base_addr && shared_page->ref_pids[i] == g_dedup_pid){
                                shared_page->ref_vas[i] = 0;
                                shared_page->ref_pids[i] = 0;
                                break;
                            }
                        }
                        if(i == MAX_SHARED_REFS) {
                            printf("Failed in userfault handle\n");
                            break;
                        }

                        if (shared_page->ref_count == 0) {
                            printf("Releasing shared page (Hash:%s)\n", shared_page->hash);
                            shared_page->valid = false;
                            g_shared_memory->shared_page_count--;
                        }
                    
                        fault_page.handle = new_handle;
                        fault_page.is_exported = false;
                        g_saved_memory -= g_page_size;
                    }
                    // printf("[%s] CoW completed for fault_addr:0x%lx base_addr:0x%lx (Hash:%s)\n", current_time().c_str(), fault_addr, base_addr, shared_page->hash);
                } else {
                    fprintf(stderr, "[%s] CoW shared page not found, fault_addr:0x%lx base_addr:0x%lx device=%d pid=%d\n", current_time().c_str(), fault_addr, base_addr, fault_page.device_id, g_dedup_pid);
                    CUmemAccessDesc accessDesc;
                    memset(&accessDesc, 0, sizeof(accessDesc));
                    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                    accessDesc.location.id = fault_page.device_id;
                    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

                    res = orig_cuMemSetAccess(base_addr, g_page_size, &accessDesc, 1);
                    if (res != CUDA_SUCCESS) {
                        printf("Failed to set access for new page: res=%d\n", res);
                        orig_cuMemUnmap(base_addr, g_page_size);
                        orig_cuMemMap(base_addr, g_page_size, 0, fault_page.handle, 0);
                    }
                }
                ret = write(fd, "done", 4);
                if (ret == -1) {
                    printf("write error, errno:%d\n", errno);
                    break;
                }
                fault_page.is_deduped = false;
                fault_page.is_stable = false;
                fault_page.unchanged_scans = 0;
                // pthread_mutex_unlock(&g_shared_memory->mutex);
                break;
            }
            default:
                printf("Unknown pagefault type in fault handler, fault_type=%d\n", info.fault_type);
                break;
        }        
    }
}

void init_userfault() {
    CUcontext ctx = nullptr;
    orig_cuCtxGetCurrent(&ctx);
    register_current_ctx_for_device();

    if (userfault_fd == -1) {
        userfault_fd = open(g_hook_config.userfault_fd_path, O_RDWR, O_CREAT);
        printf("userfault_fd %d\n", userfault_fd);
    }
    if (userfault_fd == -1) {
        perror("open userfault fd failed");
        exit(-1);
    }

    int device_id = get_current_device_id();
    if (!ensure_userfault_registration_for_current_device(device_id)) {
        exit(-1);
    }

    if (!g_userfault_handling) {
        g_userfault_handling = true;
        if (pthread_create(&userfault_handler, NULL, userfault_handler_thread, (void *)ctx)) {
            perror("pthread_create failed");
            g_userfault_handling = false;
            return;
        }
    }
}

void init_background_scan() {
    if (g_background_scanning) {
        return;
    }

    CUcontext ctx = nullptr;
    orig_cuCtxGetCurrent(&ctx);
    for (int i = 0; i < API_COUNT; ++i) {
        g_api_stats[i].calls.store(0, std::memory_order_relaxed);
        g_api_stats[i].total_ns.store(0, std::memory_order_relaxed);
    }

    if (g_hook_config.enable_background_scan) {
        g_background_scanning = true;
        if (pthread_create(&background_scan, NULL, background_scan_thread, (void *)ctx)) {
            perror("pthread_create failed for background deduplication thread");
        } else {
            printf("Background deduplication thread started\n");
        }
    } else {
        g_background_scanning = false;
        // printf("Background deduplication scan is disabled (enable_background_scan = false)\n");
    }

}

void init_shared_memory() {
    const char* SHM_NAME = g_hook_config.shared_memory_name;
    size_t shm_size = sizeof(SharedMemory);
    int shm_fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    if (shm_fd == -1) {
        perror("shm_open failed");
        exit(EXIT_FAILURE);
    }

    if (ftruncate(shm_fd, shm_size) == -1) {
        perror("ftruncate failed");
        close(shm_fd);
        exit(EXIT_FAILURE);
    }

    g_shared_memory = static_cast<SharedMemory*>(
        mmap(NULL, shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0));
    if (g_shared_memory == MAP_FAILED) {
        perror("mmap failed");
        close(shm_fd);
        exit(EXIT_FAILURE);
    }
    close(shm_fd);

    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
    pthread_mutex_init(&g_shared_memory->mutex, &attr);
    pthread_mutexattr_destroy(&attr);

    pthread_mutex_lock(&g_shared_memory->mutex);
    if (g_shared_memory->initialized != 0x12345678) {
        g_shared_memory->initialized = 0x12345678;
        g_shared_memory->shared_page_count = 0;
        g_shared_memory->candidate_page_count = 0;
        for (int i = 0; i < MAX_DEDUPE_PAGES; i++) {
            g_shared_memory->shared_pages[i].valid = false;
            g_shared_memory->candidate_pages[i].valid = false;
            memset(&g_shared_memory->shared_pages[i], 0, sizeof(SharedPageInfo));
            memset(&g_shared_memory->candidate_pages[i], 0, sizeof(CandidatePageInfo));
        }
        
        // 初始化布隆过滤器
        memset(g_shared_memory->bloom_filter, 0, BLOOM_FILTER_SIZE);
        g_shared_memory->bloom_filter_initialized = 1;
        
        printf("Shared memory initialized by first process (with bloom filter)\n");
    } else {
        printf("Attached to existing shared memory\n");
    }
    pthread_mutex_unlock(&g_shared_memory->mutex);
}

void init_hook_api() {
    void* handle = dlopen(g_hook_config.orig_path, RTLD_NOW | RTLD_NODELETE);
    if (!handle) {
        fprintf(stderr, "Failed to open libcuda.so.1: %s\n", dlerror());
        return;
    }

    #define RESOLVE_FUNC(name) \
        orig_##name = reinterpret_cast<decltype(orig_##name)>(dlsym(handle, #name)); \
        if (!orig_##name) fprintf(stderr, "Failed to resolve " #name ": %s\n", dlerror())

    orig_cuMemAlloc = (CUresult( * )(CUdeviceptr * , size_t)) dlsym(handle, "cuMemAlloc_v2");
    orig_cuMemAlloc_v2 = (CUresult( * )(CUdeviceptr * , size_t)) dlsym(handle, "cuMemAlloc_v2");
    orig_cuMemcpyHtoD = (CUresult CUDAAPI (*)(CUdeviceptr , const void* , size_t)) dlsym(handle, "cuMemcpyHtoD");
    orig_cuMemcpyDtoD = (CUresult CUDAAPI (*)(CUdeviceptr , CUdeviceptr , size_t)) dlsym(handle, "cuMemcpyDtoD_v2");
    orig_cuMemcpyDtoH = (CUresult CUDAAPI (*)(void *, CUdeviceptr, size_t)) dlsym(handle, "cuMemcpyDtoH_v2");
    orig_cuMemcpyHtoD_v2 = (CUresult CUDAAPI (*)(CUdeviceptr , const void* , size_t)) dlsym(handle, "cuMemcpyHtoD_v2");
    orig_cuMemcpyHtoDAsync = (CUresult CUDAAPI (*)(CUdeviceptr , const void* , size_t, CUstream)) dlsym(handle, "cuMemcpyHtoDAsync_v2");
    orig_cuCtxCreate = (CUresult( * )(CUcontext * , unsigned int, CUdevice)) dlsym(handle, "cuCtxCreate");
    orig_cuCtxCreate_v2 = (CUresult( * )(CUcontext * , unsigned int, CUdevice)) dlsym(handle, "cuCtxCreate_v2");
    orig_cuCtxCreate_v3 = (CUresult( * )(CUcontext * , CUexecAffinityParam * , int, unsigned int, CUdevice)) dlsym(handle, "cuCtxCreate_v3");
    orig_cuMemFree = (CUresult( * )(CUdeviceptr)) dlsym(handle, "cuMemFree_v2");

    RESOLVE_FUNC(cuGetProcAddress);
    RESOLVE_FUNC(cuGetProcAddress_v2);
    RESOLVE_FUNC(cuMemExportToShareableHandle);
    RESOLVE_FUNC(cuMemImportFromShareableHandle);
    RESOLVE_FUNC(cuMemGetAllocationGranularity);
    RESOLVE_FUNC(cuMemAddressReserve);
    RESOLVE_FUNC(cuMemAddressFree);
    RESOLVE_FUNC(cuMemSetAccess);
    RESOLVE_FUNC(cuMemCreate);
    RESOLVE_FUNC(cuMemMap);
    RESOLVE_FUNC(cuCtxGetCurrent);
    RESOLVE_FUNC(cuMemUnmap);
    RESOLVE_FUNC(cuMemRelease);
    // RESOLVE_FUNC(cuInit);
    
    #undef RESOLVE_FUNC

    if (orig_cuMemGetAllocationGranularity) {
        CUresult res = orig_cuMemGetAllocationGranularity(
            &g_page_size, nullptr, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
        if (res == CUDA_SUCCESS) {
            printf("[%s] System page size: %s\n", 
                   current_time().c_str(), 
                   format_bytes(g_page_size).c_str());
        }
    }
}

bool try_deduplicate_page(CUdeviceptr va, const void * data, size_t size, const char* current_hash) {
    //ScopedTimer t_hook(API_TRY_DEDUPLICATE_PAGE); 
    auto cur_page_it = g_page_map->find(va);
    if (cur_page_it == g_page_map->end()) {
        return false;
    }

    LocalPageInfo& cur_page = cur_page_it->second;
    if (!switch_to_device_ctx(cur_page.device_id)) {
        return false;
    }

    //为NULL的话那就只有模型参数，只读，不会写，没必要写到kernel里面
    if (current_hash != NULL) {
        update_pagefault_args args;
        args.fault_type = UVM_FAULT_TYPE_WRITE;
        args.page_addr = *(unsigned long*)&va;
        args.num_pages = 1;
        int ret = ioctl(userfault_fd, CMD_UPDATE_PAGEFAULT, (unsigned long)&args);
        if (ret == -1) {
            fprintf(stderr, "ioctl CMD_UPDATE_PAGEFAULT failed\n");
            exit(-1);
        }
    }

    //////////////////////////////////////////////////// xxh64
        // std::string hash_str;
        // {
        //     ScopedTimer t_hook(API_HASH_XXH64); 
        //     hash_str = fast_hash(data, size);
        // }
        // const char* hash = hash_str.c_str(); 
    //////////////////////////////////////////////////// xxh64 gpu
    std::string hash_str;
    {
        //ScopedTimer t_hook(API_HASH_XXH64_GPU); 
        hash_str = gpu_fast_hash_single(va, g_page_size);
    }
    const char* hash = hash_str.c_str();
    if (current_hash != NULL) {
        if (strcmp(current_hash, hash) != 0) {
            printf("[%s] hash not true in try_deduplicate_page(va=0x%llx, size=%s, current_hash=%s, hash=%s)\n", 
                current_time().c_str(), 
                (unsigned long long)va,
                format_bytes(size).c_str(),
                current_hash,
                hash);
            return false;
        }   
    }
    bool all_zero_hash = true;
    for (int i = 0; i < HASH_SIZE - 1 && hash[i] != '\0'; i++) {
        if (hash[i] != '0') {
            all_zero_hash = false;
            break;
        }
    }
    if (all_zero_hash) {
        return false;
    }

    cur_page.pagefault_flag = false;
    cur_page.is_stable = true;
    cur_page.memcpy_cnt++;
    if (cur_page.memcpy_cnt >= MAX_MEMCPY_LIMIT) 
        cur_page.is_stable = false;
    
    //pthread_mutex_lock(&g_shared_memory->mutex);

    SharedPageInfo* shared_page = nullptr;
    
    {
        //ScopedTimer t_hook(API_FIND_PAGE_BY_HASH); 
        shared_page = find_page_by_hash(hash, cur_page.dedup_domain_id);
    }

    // printf("[%s] try_deduplicate_page(va=0x%llx, size=%s, owner_va=%llx, current_hash=%s, hash=%s)\n", 
    //     current_time().c_str(), 
    //     (unsigned long long)va,
    //     format_bytes(size).c_str(),
    //     shared_page != NULL ? shared_page->owner_va : NULL, 
    //     current_hash,
    //     hash);

    if (shared_page) {
        if (shared_page->owner_pid == g_dedup_pid && shared_page->owner_va == va) {
            // fprintf(stderr, "Error : shared_page->owner_pid=%d  shared_page->owner_va=%d\n", shared_page->owner_pid, shared_page->owner_va);
            return false;
        }
        CUmemGenericAllocationHandle imported_handle;
        bool remote = g_dedup_pid != shared_page->owner_pid;
        if (!remote) {
            auto page_it = g_page_map->find(shared_page->owner_va);
            LocalPageInfo& page = page_it->second;
            imported_handle = page.handle;
            printf("[%s] va=%lx hash=%s\n",current_time().c_str(), va, hash);
        } else {
            int shareable_fd = ipc_get_shared_fd(
                shared_page->owner_pid,
                shared_page->sock_path,
                shared_page->hash,
                shared_page->device_id,
                shared_page->dedup_domain_id
            );
            
            if (shareable_fd < 0) {
                //pthread_mutex_unlock(&g_shared_memory->mutex);
                return false;
            }

            CUresult import_res;
            {
                //ScopedTimer t_hook(API_CU_MEM_IMPORT_ORIG); 
                import_res = orig_cuMemImportFromShareableHandle(
                    &imported_handle,
                    (void *)(uintptr_t)shareable_fd,
                    CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
                );
            }
            close(shareable_fd);

            if (import_res != CUDA_SUCCESS) {
                const char* errorName, *errorString;
                cuGetErrorName(import_res, &errorName);
                cuGetErrorString(import_res, &errorString);
                //pthread_mutex_unlock(&g_shared_memory->mutex);
                return false;
            }
        }
        
        bool equal = false;
        if (!remote) {
            equal = gpu_pages_equal(va, shared_page->owner_va, size);
        } else {
            CUdeviceptr alias = 0;
            if (orig_cuMemAddressReserve(&alias, size, 0, 0, 0) == CUDA_SUCCESS) {
                if (orig_cuMemMap(alias, size, 0, imported_handle, 0) == CUDA_SUCCESS) {
                    CUmemAccessDesc access = {{CU_MEM_LOCATION_TYPE_DEVICE, cur_page.device_id}, CU_MEM_ACCESS_FLAGS_PROT_READ};
                    if (orig_cuMemSetAccess(alias, size, &access, 1) == CUDA_SUCCESS)
                        equal = gpu_pages_equal(va, alias, size);
                    if (orig_cuMemUnmap(alias, size) != CUDA_SUCCESS) equal = false;
                }
                if (orig_cuMemAddressFree(alias, size) != CUDA_SUCCESS) equal = false;
            }
        }
        if (!equal) {
            if (remote) orig_cuMemRelease(imported_handle);
            return false;
        }

        {
            //ScopedTimer t_hook(API_CU_MEM_UNMAP_ORIG); 
            if (orig_cuMemUnmap(va, size) != CUDA_SUCCESS) {
                orig_cuMemRelease(imported_handle);
                //pthread_mutex_unlock(&g_shared_memory->mutex);
                return false;
            }
        }

        {
            //ScopedTimer t_hook(API_CU_MEM_MAP_ORIG); 
            if (orig_cuMemMap(va, size, 0, imported_handle, 0) != CUDA_SUCCESS) {
                orig_cuMemMap(va, size, 0, cur_page.handle, 0);
                orig_cuMemRelease(imported_handle);
                //pthread_mutex_unlock(&g_shared_memory->mutex);
                return false;
            }
        }

        CUmemAccessDesc accessDesc;
        // memset(&accessDesc, 0, sizeof(accessDesc));
        accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        accessDesc.location.id = cur_page.device_id;
        accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READ;
        {
            // printf("set access for va=%lx\n", va);
            //ScopedTimer t_hook(API_CU_MEM_SETACCESS_ORIG); 
            if (orig_cuMemSetAccess(va, size, &accessDesc, 1) != CUDA_SUCCESS) {
                // 回滚操作
                orig_cuMemUnmap(va, size);
                orig_cuMemMap(va, size, 0, cur_page.handle, 0);
                orig_cuMemRelease(imported_handle);
                //pthread_mutex_unlock(&g_shared_memory->mutex);
                return false;
            }
        }

        CUmemGenericAllocationHandle old_handle = cur_page.handle;
        cur_page.handle = imported_handle;
        cur_page.is_exported = true;
        strcpy(cur_page.hash,hash);

        {
            //ScopedTimer t_hook(API_CU_MEM_RELEASE_ORIG); 
            orig_cuMemRelease(old_handle);
        }

        g_saved_memory += size;
        g_current_allocated -= size;
        for(int i = 0; i < MAX_SHARED_REFS; i++){
            if(shared_page->ref_pids[i] == 0) {
                shared_page->ref_pids[i] = g_dedup_pid;
                shared_page->ref_vas[i] = va;
                break;
            } 
        }
        shared_page->ref_count++;

        if (current_hash != NULL) {
            printf("[] Dedup at VA: 0x%llx, Saved: %s, pid: %d, hash: %s\n",
                // current_time().c_str(),
                (unsigned long long)va,
                format_bytes(g_saved_memory).c_str(),
                getpid(),
                hash);
        } else {
            if (g_saved_memory % g_saved_memory_print_threshold == 0) 
                printf("[%s] Dedup at VA: 0x%llx, Saved: %s, pid: %d, hash: %s\n",
                    current_time().c_str(),
                    (unsigned long long)va,
                    format_bytes(g_saved_memory).c_str(),
                    getpid(),
                    hash);
        } 

        pages_deduped++;
    } else {
        ShareableHandle shareable_handle;
        CUresult export_res;
        CUmemAccessDesc accessDesc;
        memset(&accessDesc, 0, sizeof(accessDesc));
        accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        accessDesc.location.id = cur_page.device_id;
        accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READ;
        {
            // printf("set access for va=%lx\n", va);
            //ScopedTimer t_hook(API_CU_MEM_SETACCESS_ORIG); 
            if (orig_cuMemSetAccess(va, size, &accessDesc, 1) != CUDA_SUCCESS) {
                return false;
            }
        }

        if (cur_page.is_exported == false) {
            //ScopedTimer t_hook(API_CU_MEM_EXPORT_ORIG); 
            export_res = orig_cuMemExportToShareableHandle(
                (void*)&shareable_handle,
                cur_page.handle,
                CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                0
            );
            if (export_res == CUDA_SUCCESS) {
                cur_page.is_exported = true;
                int free_index = -1;
                for (int i = 0; i < MAX_DEDUPE_PAGES; i++) {
                    if (!g_shared_memory->shared_pages[i].valid) {
                        free_index = i;
                        break;
                    }
                }
                
                if (free_index == -1) {
                    fprintf(stderr, "Global dedupe page storage full!\n");
                    //pthread_mutex_unlock(&g_shared_memory->mutex);
                    return false;
                }
                
                SharedPageInfo* new_shared_page = &g_shared_memory->shared_pages[free_index];
                memset(new_shared_page, 0, sizeof(SharedPageInfo)); 
                
                strncpy(new_shared_page->hash, hash, HASH_SIZE - 1);
                new_shared_page->hash[HASH_SIZE - 1] = '\0';
                new_shared_page->owner_pid = g_dedup_pid;
                strncpy(new_shared_page->sock_path, 
                        ipc_get_current_sock_path().c_str(), 
                        sizeof(new_shared_page->sock_path)-1);
                new_shared_page->ref_count = 1;
                new_shared_page->valid = true;
                new_shared_page->device_id = cur_page.device_id;
                new_shared_page->dedup_domain_id = cur_page.dedup_domain_id;
                new_shared_page->owner_va = va;
                g_shared_memory->shared_page_count++;
                strncpy(cur_page.hash, hash, HASH_SIZE);

                // printf("[%s] export handle va=%lx hash=%s handle=%lx\n", current_time().c_str(), va, hash, cur_page.handle);
            } else {
                const char* errorName, *errorString;
                cuGetErrorName(export_res, &errorName);
                cuGetErrorString(export_res, &errorString);
                fprintf(stderr, "[%s] Export failed: %s (%s) va=%lx hash=%s handle=%lx cur_page.is_exported=%d pid=%d\n", current_time().c_str(), errorName, errorString, va, hash, cur_page.handle, cur_page.is_exported, g_dedup_pid);
            }
        } else {
            cur_page.is_exported = true;
            int free_index = -1;
            for (int i = 0; i < MAX_DEDUPE_PAGES; i++) {
                if (!g_shared_memory->shared_pages[i].valid) {
                    free_index = i;
                    break;
                }
            }
            
            if (free_index == -1) {
                fprintf(stderr, "Global dedupe page storage full!\n");
                //pthread_mutex_unlock(&g_shared_memory->mutex);
                return false;
            }
            
            SharedPageInfo* new_shared_page = &g_shared_memory->shared_pages[free_index];
            memset(new_shared_page, 0, sizeof(SharedPageInfo)); 
            
            strncpy(new_shared_page->hash, hash, HASH_SIZE - 1);
            new_shared_page->hash[HASH_SIZE - 1] = '\0';
            new_shared_page->owner_pid = g_dedup_pid;
            strncpy(new_shared_page->sock_path, 
                    ipc_get_current_sock_path().c_str(), 
                    sizeof(new_shared_page->sock_path)-1);
            new_shared_page->ref_count = 1;
            new_shared_page->valid = true;
            new_shared_page->device_id = cur_page.device_id;
            new_shared_page->dedup_domain_id = cur_page.dedup_domain_id;
            new_shared_page->owner_va = va;
            g_shared_memory->shared_page_count++;

            strncpy(cur_page.hash, hash, HASH_SIZE);
        }
    }

    for (int i = 0; i < MAX_DEDUPE_PAGES; i++) {
        if (g_shared_memory->candidate_pages[i].valid &&
            g_shared_memory->candidate_pages[i].dedup_domain_id == cur_page.dedup_domain_id &&
            strncmp(g_shared_memory->candidate_pages[i].hash, hash, HASH_SIZE) == 0) {
            g_shared_memory->candidate_pages[i].valid = false;
            g_shared_memory->candidate_page_count--;
        }
    }
    // pthread_mutex_unlock(&g_shared_memory->mutex);
    return true;
}

extern "C" {
CUresult CUDAAPI cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytesize) {
    ScopedTimer t_hook(API_CU_MEM_ALLOC_HOOK); 
    if (g_shared_memory == nullptr) {
        printf("Initializing shared memory...\n");
        init_shared_memory();
        printf("Shared memory initialized at %p\n", (void*)g_shared_memory);
    }

    register_current_ctx_for_device();
    if (userfault_fd == -1) {
        printf("Initializing userfault...\n");
    }
    init_userfault();
    if (!g_background_scanning) {
        // printf("userfault initialized\n");
        init_background_scan();
    }

    CUresult res = CUDA_ERROR_UNKNOWN;
    if(bytesize < g_page_size) {
        ScopedTimer t_hook(API_CU_MEM_ALLOC_ORIG); 
        res = orig_cuMemAlloc? 
                    orig_cuMemAlloc(dptr, bytesize) : 
                    CUDA_ERROR_NOT_INITIALIZED;
        if (res != CUDA_SUCCESS) {
            printf("hook in cuMemAlloc, res %d\n",res);
        }
        return res;
    }
    int cur_dev = get_current_device_id();
    if (cur_dev < 0) {
        fprintf(stderr, "cuMemAlloc_v2: failed to get current device id\n");
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    int cur_dedup_domain = get_current_dedup_domain_id();
    
    CUmemAllocationProp prop;
    memset(&prop, 0, sizeof(prop));
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = cur_dev;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    
    size_t aligned_size = ((bytesize + g_page_size - 1) / g_page_size) * g_page_size;
    size_t num_pages = aligned_size / g_page_size;
    VMMAllocInfo alloc_info;
    alloc_info.aligned_size = aligned_size;
    alloc_info.device_id = cur_dev;
    alloc_info.dedup_domain_id = cur_dedup_domain;
    
    CUdeviceptr va;
    {
        ScopedTimer t_orig(API_CU_MEM_ADDRESS_RESERVE_ORIG);
        res = orig_cuMemAddressReserve(&va, aligned_size, 0, 0, 0);
    }
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "cuMemAddressReserve failed: %d\n", res);
        return res;
    }

    for (size_t i = 0; i < num_pages; i++) {
        CUmemGenericAllocationHandle handle;
        CUdeviceptr va_ptr = va + i * g_page_size;

        {
            ScopedTimer t_hook_create(API_CU_MEM_CREATE_HOOK);
            res = cuMemCreate(&handle, g_page_size, &prop, 0);
        }
        if (res != CUDA_SUCCESS) {
            fprintf(stderr, "cuMemCreate failed: %d\n", res);
            return res;
        }

        {
            ScopedTimer t_orig_map(API_CU_MEM_MAP_ORIG);
            res = orig_cuMemMap(va_ptr, g_page_size, 0, handle, 0);
        }
        if (res != CUDA_SUCCESS) {
            fprintf(stderr, "cuMemMap failed: %d\n", res);
            orig_cuMemRelease(handle);
            return res;
        }
        LocalPageInfo page;
        page.handle = handle;
        page.va = va_ptr;
        page.device_id = cur_dev;
        page.dedup_domain_id = cur_dedup_domain;
        memset(page.hash, 0, HASH_SIZE);
        alloc_info.pages.push_back(page);
        (*g_page_map)[va_ptr] = page;
    }

    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = cur_dev;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    
    {
    ScopedTimer t_orig_setaccess(API_CU_MEM_SETACCESS_ORIG);
    res = orig_cuMemSetAccess(va, aligned_size, &accessDesc, 1);
    }
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "cuMemSetAccess failed: %d\n", res);
        orig_cuMemUnmap(va, aligned_size);
        return res;
    }

    alloc_info.va = va;
    g_allocations_vmm[va] = alloc_info;
    
    *dptr = va;
    printf("[%s] cuMemAlloc_v2(dptr=0x%llx, aligned_size=%s)\n", 
        current_time().c_str(), 
        (unsigned long long)*dptr,
        format_bytes(aligned_size).c_str());

    return CUDA_SUCCESS;
}

// CUresult CUDAAPI cuMemAlloc(CUdeviceptr* dptr, size_t bytesize) {
//     ScopedTimer t_hook(API_CU_MEM_ALLOC_HOOK); 
//     if (userfault_fd == -1) {
//         printf("Initializing shared memory...\n");
//         init_shared_memory();
//         printf("Shared memory initialized at %p\n", (void*)g_shared_memory);
        
//         printf("Initializing userfault...\n");
//         init_userfault();
//         printf("userfault initialized\n");

//         init_background_scan();
//     }

//     CUresult res = CUDA_ERROR_UNKNOWN;
//     if(bytesize < g_page_size) {
//         ScopedTimer t_hook(API_CU_MEM_ALLOC_ORIG); 
//         res = orig_cuMemAlloc? 
//                     orig_cuMemAlloc(dptr, bytesize) : 
//                     CUDA_ERROR_NOT_INITIALIZED;
//         if (res != CUDA_SUCCESS) 
//             printf("hook in cuMemAlloc, res %d\n",res);
        
//         return res;
//     }
//     CUcontext ctx;    
    
//     CUmemAllocationProp prop;
//     memset(&prop, 0, sizeof(prop));
//     prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
//     prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
//     prop.location.id = DEDUP_DEVICE;
//     prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    
//     size_t aligned_size = ((bytesize + g_page_size - 1) / g_page_size) * g_page_size;
//     size_t num_pages = aligned_size / g_page_size;
//     VMMAllocInfo alloc_info;
//     alloc_info.aligned_size = aligned_size;
    
//     CUdeviceptr va;
//     {
//         ScopedTimer t_orig(API_CU_MEM_ADDRESS_RESERVE_ORIG);
//         res = orig_cuMemAddressReserve(&va, aligned_size, 0, 0, 0);
//     }
//     if (res != CUDA_SUCCESS) {
//         fprintf(stderr, "cuMemAddressReserve failed: %d\n", res);
//         return res;
//     }

//     //////////////////////////////////////////////////
//     // 延迟分配
//     //////////////////////////////////////////////////
//     update_pagefault_args args;
//     args.fault_type = UVM_FAULT_TYPE_INVALID_PDE;
//     args.page_addr = *(unsigned long*)&va;
//     args.num_pages = num_pages;
//     int ret = ioctl(userfault_fd, CMD_UPDATE_PAGEFAULT, (unsigned long)&args);

//     for (size_t i = 0; i < num_pages; i++) {
//         CUdeviceptr va_ptr = va + i * g_page_size;
//         LocalPageInfo page;
//         page.handle = 0;
//         page.va = va_ptr;
//         memset(page.hash, 0, HASH_SIZE);
//         alloc_info.pages.push_back(page);
//         (*g_page_map)[va_ptr] = page;
//     }

//     alloc_info.va = va;
//     g_allocations_vmm[va] = alloc_info;
//     return CUDA_SUCCESS;
//     //////////////////////////////////////////////////
//     // old
//     //////////////////////////////////////////////////

//     for (size_t i = 0; i < num_pages; i++) {
//         CUmemGenericAllocationHandle handle;
//         CUdeviceptr va_ptr = va + i * g_page_size;
//         {
//             ScopedTimer t_hook_create(API_CU_MEM_CREATE_HOOK);
//             res = cuMemCreate(&handle, g_page_size, &prop, 0);
//         }
//         if (res != CUDA_SUCCESS) {
//             fprintf(stderr, "cuMemCreate failed: %d\n", res);
//             return res;
//         }

//         {
//             ScopedTimer t_orig_map(API_CU_MEM_MAP_ORIG);
//             res = orig_cuMemMap(va_ptr, g_page_size, 0, handle, 0);
//         }
//         if (res != CUDA_SUCCESS) {
//             fprintf(stderr, "cuMemMap failed: %d\n", res);
//             orig_cuMemRelease(handle);
//             return res;
//         }
//         LocalPageInfo page;
//         page.handle = handle;
//         page.va = va_ptr;
//         memset(page.hash, 0, HASH_SIZE);
//         alloc_info.pages.push_back(page);
//         (*g_page_map)[va_ptr] = page;
//     }

//     CUmemAccessDesc accessDesc = {};
//     accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
//     accessDesc.location.id = DEDUP_DEVICE;
//     accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    
//     {
//         ScopedTimer t_orig_setaccess(API_CU_MEM_SETACCESS_ORIG);
//         res = orig_cuMemSetAccess(va, aligned_size, &accessDesc, 1);
//     }
//     if (res != CUDA_SUCCESS) {
//         fprintf(stderr, "cuMemSetAccess failed: %d\n", res);
//         orig_cuMemUnmap(va, aligned_size);
//         return res;
//     }

//     alloc_info.va = va;
//     g_allocations_vmm[va] = alloc_info;
    
//     *dptr = va;
//     g_current_allocated += aligned_size;
//     printf("[%s] cuMemAlloc(dptr=0x%llx, aligned_size=%s, g_current_allocated=%s)\n", 
//         current_time().c_str(), 
//         (unsigned long long)*dptr,
//         format_bytes(aligned_size).c_str(),
//         format_bytes(g_current_allocated).c_str());

//     return CUDA_SUCCESS;
// }

CUresult CUDAAPI cuMemAlloc(CUdeviceptr* dptr, size_t bytesize) {
    ScopedTimer t_hook(API_CU_MEM_ALLOC_HOOK); 
    if (g_shared_memory == nullptr) {
        printf("Initializing shared memory...\n");
        init_shared_memory();
        printf("Shared memory initialized at %p\n", (void*)g_shared_memory);
    }

    register_current_ctx_for_device();
    if (userfault_fd == -1) {
        printf("Initializing userfault...\n");
    }
    init_userfault();
    if (!g_background_scanning) {
        // printf("userfault initialized\n");
        init_background_scan();
    }

    CUresult res = CUDA_ERROR_UNKNOWN;
    if(bytesize < g_page_size) {
        ScopedTimer t_hook(API_CU_MEM_ALLOC_ORIG); 
        res = orig_cuMemAlloc? 
                    orig_cuMemAlloc(dptr, bytesize) : 
                    CUDA_ERROR_NOT_INITIALIZED;
        if (res != CUDA_SUCCESS) 
            printf("hook in cuMemAlloc, res %d\n",res);
        
        return res;
    }
    int cur_dev = get_current_device_id();
    if (cur_dev < 0) {
        fprintf(stderr, "cuMemAlloc: failed to get current device id\n");
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    int cur_dedup_domain = get_current_dedup_domain_id();
    
    CUmemAllocationProp prop;
    memset(&prop, 0, sizeof(prop));
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = cur_dev;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    
    size_t aligned_size = ((bytesize + g_page_size - 1) / g_page_size) * g_page_size;
    size_t num_pages = aligned_size / g_page_size;
    VMMAllocInfo alloc_info;
    alloc_info.aligned_size = aligned_size;
    alloc_info.device_id = cur_dev;
    alloc_info.dedup_domain_id = cur_dedup_domain;
    
    CUdeviceptr va;
    {
        ScopedTimer t_orig(API_CU_MEM_ADDRESS_RESERVE_ORIG);
        res = orig_cuMemAddressReserve(&va, aligned_size, 0, 0, 0);
    }
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "cuMemAddressReserve failed: %d\n", res);
        return res;
    }

    for (size_t i = 0; i < num_pages; i++) {
        CUmemGenericAllocationHandle handle;
        CUdeviceptr va_ptr = va + i * g_page_size;
        {
            ScopedTimer t_hook_create(API_CU_MEM_CREATE_HOOK);
            res = cuMemCreate(&handle, g_page_size, &prop, 0);
        }
        if (res != CUDA_SUCCESS) {
            fprintf(stderr, "cuMemCreate failed: %d\n", res);
            return res;
        }

        {
            ScopedTimer t_orig_map(API_CU_MEM_MAP_ORIG);
            res = orig_cuMemMap(va_ptr, g_page_size, 0, handle, 0);
        }
        if (res != CUDA_SUCCESS) {
            fprintf(stderr, "cuMemMap failed: %d\n", res);
            orig_cuMemRelease(handle);
            return res;
        }
        LocalPageInfo page;
        page.handle = handle;
        page.va = va_ptr;
        page.device_id = cur_dev;
        page.dedup_domain_id = cur_dedup_domain;
        memset(page.hash, 0, HASH_SIZE);
        alloc_info.pages.push_back(page);
        (*g_page_map)[va_ptr] = page;
    }

    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = cur_dev;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    
    {
        ScopedTimer t_orig_setaccess(API_CU_MEM_SETACCESS_ORIG);
        res = orig_cuMemSetAccess(va, aligned_size, &accessDesc, 1);
    }
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "cuMemSetAccess failed: %d\n", res);
        orig_cuMemUnmap(va, aligned_size);
        return res;
    }

    alloc_info.va = va;
    g_allocations_vmm[va] = alloc_info;
    
    *dptr = va;
    g_current_allocated += aligned_size;
    // printf("[%s] cuMemAlloc(dptr=0x%llx, aligned_size=%s, g_current_allocated=%s)\n", 
    //     current_time().c_str(), 
    //     (unsigned long long)*dptr,
    //     format_bytes(aligned_size).c_str(),
    //     format_bytes(g_current_allocated).c_str());

    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount) {
    // 小于2MB的分配直接由原生API处理
    ScopedTimer t_hook(API_CU_MEMCPY_HTOD_ASYNC_HOOK); 
    if(ByteCount < g_page_size) {
        CUresult res;
        res = orig_cuMemcpyHtoD_v2? 
                    orig_cuMemcpyHtoD_v2(dstDevice, srcHost, ByteCount) : 
                    CUDA_ERROR_NOT_INITIALIZED;
        if (res != CUDA_SUCCESS) 
            printf("hook in cuMemcpyHtoD, res=%d dstDevice=%lx\n", res, dstDevice);
        return res;
    }

    g_cuMemCpy_size += ByteCount;

    // printf("[%s] cuMemcpyHtoD_v2(dst=0x%llx, size=%s, total_size=%s)\n", 
    //        current_time().c_str(), 
    //        (unsigned long long)dstDevice,
    //        format_bytes(ByteCount).c_str(),
    //        format_bytes(g_cuMemCpy_size).c_str());
    
    CUdeviceptr current_addr = dstDevice;
    int bytes_remaining = ByteCount;
    const char* current_src = static_cast<const char*>(srcHost);
    
    while (bytes_remaining > 0) {
        bool in_vmm = false;
        CUdeviceptr page_start = 0;
        size_t page_offset = 0;
        size_t page_bytes = 0;
        int page_device_id = -1;
        
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            auto page_it = g_page_map->lower_bound(current_addr);
            if (page_it != g_page_map->end() && current_addr == page_it->first){
                page_offset = 0;
                page_bytes = std::min((int)g_page_size, bytes_remaining);
                page_start = page_it->first;
                page_device_id = page_it->second.device_id;
                in_vmm = true;
            }
            if (page_it != g_page_map->begin()) {
                page_it--;
                if (current_addr > page_it->first && current_addr < page_it->first + g_page_size){
                    page_offset = current_addr - page_it->first;
                    page_bytes = g_page_size - page_offset;
                    page_start = page_it->first;
                    page_device_id = page_it->second.device_id;
                    in_vmm = true;
                }
            }
        }

        if (!in_vmm) {
            CUresult res;
            {
                ScopedTimer t_hook(API_CU_MEMCPY_HTOD_ORIG); 
                res = orig_cuMemcpyHtoD_v2? 
                            orig_cuMemcpyHtoD_v2(current_addr, static_cast<const void*>(current_src), g_page_size) : 
                            CUDA_ERROR_NOT_INITIALIZED;
            }
            if (res != CUDA_SUCCESS) {
                printf("hook in cuMemcpyHtoD, res=%d current_addr=%lx\n", res, current_addr);
                return res;
            }

            size_t skip_bytes = std::min(
                (unsigned long long)bytes_remaining,
                g_page_size - (current_addr % g_page_size)
            );

            current_addr += skip_bytes;
            current_src += skip_bytes;
            bytes_remaining -= skip_bytes;
            continue;
        }
        
        CUresult res;
        {
            ScopedTimer t_hook(API_CU_MEMCPY_HTOD_ORIG); 
            res = orig_cuMemcpyHtoD_v2? 
                        orig_cuMemcpyHtoD_v2(current_addr, static_cast<const void*>(current_src), page_bytes) : 
                        CUDA_ERROR_NOT_INITIALIZED;
        }
        if (res != CUDA_SUCCESS) {
            printf("hook in cuMemcpyHtoD, res=%d current_addr=%lx\n", res, current_addr);
            if (res == 1) {
                CUmemAccessDesc accessDesc;
                memset(&accessDesc, 0, sizeof(accessDesc));
                accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                accessDesc.location.id = page_device_id;
                accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                {
                    ScopedTimer t_hook(API_CU_MEM_SETACCESS_ORIG);
                    res = orig_cuMemSetAccess(page_start, g_page_size, &accessDesc, 1);
                }
                if (res != CUDA_SUCCESS) {
                    printf("cuMemSetAccess fail, res:%d\n", res);
                    return res;
                }
                {
                    ScopedTimer t_hook(API_CU_MEMCPY_HTOD_ORIG); 
                    res = orig_cuMemcpyHtoD_v2? 
                        orig_cuMemcpyHtoD_v2(current_addr, static_cast<const void*>(current_src), page_bytes) : 
                        CUDA_ERROR_NOT_INITIALIZED;
                }
                if (res != CUDA_SUCCESS)
                    printf("hook in cuMemcpyHtoD, res=%d current_addr=%lx\n", res, current_addr);
            } else return res;
        }
        
        if (page_offset == 0 && page_bytes == g_page_size) {
            LocalPageInfo& page = (g_page_map->find(page_start))->second;
            if (page.memcpy_cnt < MAX_MEMCPY_LIMIT) {
                if (try_deduplicate_page(page_start, srcHost, g_page_size, NULL)) {
                    page.is_stable = true;
                    page.is_deduped = true;
                }
            }
        }

        current_addr += page_bytes;
        current_src += page_bytes;
        bytes_remaining -= page_bytes;
    }
    
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyHtoD(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount) {
    // 小于2MB的分配直接由原生API处理
    ScopedTimer t_hook(API_CU_MEMCPY_HTOD_ASYNC_HOOK); 
    if(ByteCount < g_page_size) {
        CUresult res;
        res = orig_cuMemcpyHtoD_v2? 
                    orig_cuMemcpyHtoD_v2(dstDevice, srcHost, ByteCount) : 
                    CUDA_ERROR_NOT_INITIALIZED;
        if (res != CUDA_SUCCESS) 
            printf("hook in cuMemcpyHtoD, res=%d dstDevice=%lx\n", res, dstDevice);
        return res;
    }

    g_cuMemCpy_size += ByteCount;

    // printf("[%s] cuMemcpyHtoD(dst=0x%llx, size=%s, total_size=%s)\n", 
    //        current_time().c_str(), 
    //        (unsigned long long)dstDevice,
    //        format_bytes(ByteCount).c_str(),
    //        format_bytes(g_cuMemCpy_size).c_str());
    
    CUdeviceptr current_addr = dstDevice;
    int bytes_remaining = ByteCount;
    const char* current_src = static_cast<const char*>(srcHost);
    
    while (bytes_remaining > 0) {
        bool in_vmm = false;
        CUdeviceptr page_start = 0;
        size_t page_offset = 0;
        size_t page_bytes = 0;
        int page_device_id = -1;
        
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            auto page_it = g_page_map->lower_bound(current_addr);
            if (page_it != g_page_map->end() && current_addr == page_it->first){
                page_offset = 0;
                page_bytes = std::min((int)g_page_size, bytes_remaining);
                page_start = page_it->first;
                page_device_id = page_it->second.device_id;
                in_vmm = true;
            }
            if (page_it != g_page_map->begin()) {
                page_it--;
                if (current_addr > page_it->first && current_addr < page_it->first + g_page_size){
                    page_offset = current_addr - page_it->first;
                    page_bytes = g_page_size - page_offset;
                    page_start = page_it->first;
                    page_device_id = page_it->second.device_id;
                    in_vmm = true;
                }
            }
        }

        if (!in_vmm) {
            CUresult res;
            {
                ScopedTimer t_hook(API_CU_MEMCPY_HTOD_ORIG); 
                res = orig_cuMemcpyHtoD_v2? 
                            orig_cuMemcpyHtoD_v2(current_addr, static_cast<const void*>(current_src), g_page_size) : 
                            CUDA_ERROR_NOT_INITIALIZED;
            }
            if (res != CUDA_SUCCESS) {
                printf("hook in cuMemcpyHtoD, res=%d current_addr=%lx\n", res, current_addr);
                return res;
            }

            size_t skip_bytes = std::min(
                (unsigned long long)bytes_remaining,
                g_page_size - (current_addr % g_page_size)
            );

            current_addr += skip_bytes;
            current_src += skip_bytes;
            bytes_remaining -= skip_bytes;
            continue;
        }
        
        CUresult res;
        {
            ScopedTimer t_hook(API_CU_MEMCPY_HTOD_ORIG); 
            res = orig_cuMemcpyHtoD_v2? 
                        orig_cuMemcpyHtoD_v2(current_addr, static_cast<const void*>(current_src), page_bytes) : 
                        CUDA_ERROR_NOT_INITIALIZED;
        }
        if (res != CUDA_SUCCESS) {
            printf("hook in cuMemcpyHtoD, res=%d current_addr=%lx\n", res, current_addr);
            if (res == 1) {
                CUmemAccessDesc accessDesc;
                memset(&accessDesc, 0, sizeof(accessDesc));
                accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                accessDesc.location.id = page_device_id;
                accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                {
                    ScopedTimer t_hook(API_CU_MEM_SETACCESS_ORIG);
                    res = orig_cuMemSetAccess(page_start, g_page_size, &accessDesc, 1);
                }
                if (res != CUDA_SUCCESS) {
                    printf("cuMemSetAccess fail, res:%d\n", res);
                    return res;
                }
                {
                    ScopedTimer t_hook(API_CU_MEMCPY_HTOD_ORIG); 
                    res = orig_cuMemcpyHtoD_v2? 
                        orig_cuMemcpyHtoD_v2(current_addr, static_cast<const void*>(current_src), page_bytes) : 
                        CUDA_ERROR_NOT_INITIALIZED;
                }
                if (res != CUDA_SUCCESS)
                    printf("hook in cuMemcpyHtoD, res=%d current_addr=%lx\n", res, current_addr);
            } else return res;
        }
        
        if (page_offset == 0 && page_bytes == g_page_size) {
            LocalPageInfo& page = (g_page_map->find(page_start))->second;
            if (page.memcpy_cnt < MAX_MEMCPY_LIMIT) {
                if (try_deduplicate_page(page_start, srcHost, g_page_size, NULL)) {
                    page.is_stable = true;
                    page.is_deduped = true;
                }
            }
        }

        current_addr += page_bytes;
        current_src += page_bytes;
        bytes_remaining -= page_bytes;
    }
    
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream) {
    // 小于2MB的分配直接由原生API处理
    ScopedTimer t_hook(API_CU_MEMCPY_HTOD_ASYNC_HOOK); 
    if(ByteCount < g_page_size) {
        CUresult res;
        res = orig_cuMemcpyHtoD_v2? 
                    orig_cuMemcpyHtoD_v2(dstDevice, srcHost, ByteCount) : 
                    CUDA_ERROR_NOT_INITIALIZED;
        if (res != CUDA_SUCCESS) 
            printf("hook in cuMemcpyHtoD, res=%d dstDevice=%lx\n", res, dstDevice);
        return res;
    }

    g_cuMemCpy_size += ByteCount;

    // printf("[%s] cuMemcpyHtoDAsync(dst=0x%llx, size=%s, total_size=%s)\n", 
    //        current_time().c_str(), 
    //        (unsigned long long)dstDevice,
    //        format_bytes(ByteCount).c_str(),
    //        format_bytes(g_cuMemCpy_size).c_str());
    
    CUdeviceptr current_addr = dstDevice;
    int bytes_remaining = ByteCount;
    const char* current_src = static_cast<const char*>(srcHost);
    
    while (bytes_remaining > 0) {
        bool in_vmm = false;
        CUdeviceptr page_start = 0;
        size_t page_offset = 0;
        size_t page_bytes = 0;
        int page_device_id = -1;
        
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            auto page_it = g_page_map->lower_bound(current_addr);
            if (page_it != g_page_map->end() && current_addr == page_it->first){
                page_offset = 0;
                page_bytes = std::min((int)g_page_size, bytes_remaining);
                page_start = page_it->first;
                page_device_id = page_it->second.device_id;
                in_vmm = true;
            }
            if (page_it != g_page_map->begin()) {
                page_it--;
                if (current_addr > page_it->first && current_addr < page_it->first + g_page_size){
                    page_offset = current_addr - page_it->first;
                    page_bytes = g_page_size - page_offset;
                    page_start = page_it->first;
                    page_device_id = page_it->second.device_id;
                    in_vmm = true;
                }
            }
        }

        if (!in_vmm) {
            CUresult res;
            {
                ScopedTimer t_hook(API_CU_MEMCPY_HTOD_ORIG); 
                res = orig_cuMemcpyHtoD_v2? 
                            orig_cuMemcpyHtoD_v2(current_addr, static_cast<const void*>(current_src), g_page_size) : 
                            CUDA_ERROR_NOT_INITIALIZED;
            }
            if (res != CUDA_SUCCESS) {
                printf("hook in cuMemcpyHtoD, res=%d current_addr=%lx\n", res, current_addr);
                return res;
            }

            size_t skip_bytes = std::min(
                (unsigned long long)bytes_remaining,
                g_page_size - (current_addr % g_page_size)
            );

            current_addr += skip_bytes;
            current_src += skip_bytes;
            bytes_remaining -= skip_bytes;
            continue;
        }
        
        CUresult res;
        {
            ScopedTimer t_hook(API_CU_MEMCPY_HTOD_ORIG); 
            res = orig_cuMemcpyHtoD_v2? 
                        orig_cuMemcpyHtoD_v2(current_addr, static_cast<const void*>(current_src), page_bytes) : 
                        CUDA_ERROR_NOT_INITIALIZED;
        }
        if (res != CUDA_SUCCESS) {
            printf("hook in cuMemcpyHtoD, res=%d current_addr=%lx\n", res, current_addr);
            if (res == 1) {
                CUmemAccessDesc accessDesc;
                memset(&accessDesc, 0, sizeof(accessDesc));
                accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                accessDesc.location.id = page_device_id;
                accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                {
                    ScopedTimer t_hook(API_CU_MEM_SETACCESS_ORIG);
                    res = orig_cuMemSetAccess(page_start, g_page_size, &accessDesc, 1);
                }
                if (res != CUDA_SUCCESS) {
                    printf("cuMemSetAccess fail, res:%d\n", res);
                    return res;
                }
                {
                    ScopedTimer t_hook(API_CU_MEMCPY_HTOD_ORIG); 
                    res = orig_cuMemcpyHtoD_v2? 
                        orig_cuMemcpyHtoD_v2(current_addr, static_cast<const void*>(current_src), page_bytes) : 
                        CUDA_ERROR_NOT_INITIALIZED;
                }
                if (res != CUDA_SUCCESS)
                    printf("hook in cuMemcpyHtoD, res=%d current_addr=%lx\n", res, current_addr);
            } else return res;
        }
        
        if (page_offset == 0 && page_bytes == g_page_size) {
            LocalPageInfo& page = (g_page_map->find(page_start))->second;
            if (page.memcpy_cnt < MAX_MEMCPY_LIMIT) {
                if (try_deduplicate_page(page_start, srcHost, g_page_size, NULL)) {
                    page.is_stable = true;
                    page.is_deduped = true;
                }
            }
        }

        current_addr += page_bytes;
        current_src += page_bytes;
        bytes_remaining -= page_bytes;
    }
    
    return CUDA_SUCCESS;
}

CUresult cuMemFree(CUdeviceptr dptr) {
    CUresult res = CUDA_ERROR_UNKNOWN;

    auto it = g_allocations_vmm.find(dptr);
    if (it == g_allocations_vmm.end()) 
        return orig_cuMemFree(dptr);

    VMMAllocInfo& alloc_info = it->second;
    int num_pages = alloc_info.pages.size();
    for (size_t i = 0; i < num_pages; i++) {
        LocalPageInfo p = alloc_info.pages[i];
        CUdeviceptr va_ptr = alloc_info.va + i * g_page_size;
        g_page_map->erase(va_ptr);
        CUresult res1 = orig_cuMemUnmap(va_ptr, g_page_size);
        CUresult res2 = orig_cuMemRelease(p.handle);
        if (res1 != CUDA_SUCCESS || res2 != CUDA_SUCCESS) {
            fprintf(stderr, "cuMemFree: page free error: 0x%llx\n", (unsigned long long)va_ptr);
            return CUDA_ERROR_INVALID_VALUE;
        }
    } 
    res = orig_cuMemAddressFree(alloc_info.va, alloc_info.aligned_size);
    printf("[%s] cuMemFree(dptr=0x%llx ,size=%s)\n", 
        current_time().c_str(), 
        (unsigned long long)dptr,
        format_bytes(alloc_info.aligned_size).c_str());
    g_allocations_vmm.erase(it);
    return res;
}

CUresult CUDAAPI cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream) {
    CUresult res = CUDA_ERROR_UNKNOWN;

    auto it = g_allocations_vmm.find(dptr);
    if (it == g_allocations_vmm.end()) 
        return orig_cuMemFree(dptr);

    printf("[%s] cuMemFreeAsync(dptr=0x%llx)\n", 
        current_time().c_str(), 
        (unsigned long long)dptr);

    VMMAllocInfo& alloc_info = it->second;
    int num_pages = alloc_info.pages.size();
    for (size_t i = 0; i < num_pages; i++) {
        LocalPageInfo p = alloc_info.pages[i];
        CUdeviceptr va_ptr = alloc_info.va + i * g_page_size;
        g_page_map->erase(va_ptr);
        CUresult res1 = orig_cuMemUnmap(va_ptr, g_page_size);
        CUresult res2 = orig_cuMemRelease(p.handle);
        if (res1 != CUDA_SUCCESS || res2 != CUDA_SUCCESS) {
            fprintf(stderr, "cuMemFreeAsync: page free error: 0x%llx\n", (unsigned long long)va_ptr);
            return CUDA_ERROR_INVALID_VALUE;
        }
    }
    res = orig_cuMemAddressFree(alloc_info.va, alloc_info.aligned_size);
    g_allocations_vmm.erase(it);
    return res;
}

void *local_handle = NULL;
CUresult CUDAAPI cuGetProcAddress_v2(
    const char* symbol, void** pfn, int  cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult* symbolStatus
) { 
    if (!local_handle)
	    local_handle = dlopen (g_hook_config.hook_path, RTLD_NOW | RTLD_NODELETE);
    *pfn = NULL;
    *pfn = dlsym (local_handle, symbol);
    CUresult res = CUDA_ERROR_UNKNOWN;
    if (*pfn)
	    res = CUDA_SUCCESS;
    else
	    res = orig_cuGetProcAddress_v2 (symbol, pfn, cudaVersion, flags, symbolStatus);
    // printf("[%s] cuGetProcAddress_v2 called for: %s\n", current_time().c_str(), symbol);
    
    return res;
}

CUresult CUDAAPI cuGetProcAddress(
    const char* symbol, void** pfn, int  cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult* symbolStatus
) {
    CUresult res = CUDA_ERROR_UNKNOWN;
    if (!strcmp (symbol, "cuMemAlloc") || !strcmp (symbol, "cuMemAlloc_v2") || 
    !strncmp (symbol, "cuMemcpyHtoD", 12) || !strncmp (symbol, "cuGetProcAddress", 16) ||
    !strcmp (symbol, "cuMemCreate") || !strcmp (symbol, "cuMemMap") || !strncmp (symbol, "cuMemFree", 9)) {
        if (!local_handle)
	        local_handle = dlopen (g_hook_config.hook_path, RTLD_NOW | RTLD_NODELETE);
        *pfn = NULL;
        *pfn = dlsym (local_handle, symbol);
        // printf("[%s] cuGetProcAddress called for: %s\n", current_time().c_str(), symbol);
        if (*pfn)
            res = CUDA_SUCCESS;  
    } else {
        if (orig_cuGetProcAddress) {
            res = orig_cuGetProcAddress(symbol, pfn, cudaVersion, flags, symbolStatus);
        }
    }
    return res;
}

// cuMemCreate hook 
CUresult CUDAAPI cuMemCreate(CUmemGenericAllocationHandle* handle, size_t size, const CUmemAllocationProp* prop, unsigned long long flags) {
    size_t aligned_size = ((size + g_page_size - 1) / g_page_size) * g_page_size;
    
    CUresult res = orig_cuMemCreate ? 
                   orig_cuMemCreate(handle, aligned_size, prop, flags) : 
                   CUDA_ERROR_NOT_INITIALIZED;
    
    if (res == CUDA_SUCCESS) {
        size_t num_pages = aligned_size / g_page_size;
        g_current_allocated += aligned_size;
    }
    
    return res;
}

} // extern "C"

__attribute__((destructor)) void cleanup() {
    if (g_shared_memory) {
        printf("Cleaning up shared memory...\n");
        munmap(g_shared_memory, sizeof(SharedMemory));
        shm_unlink("/cuda_dedup_shared");
    }
    printf("close userfault_fd\n");
    close(userfault_fd);
    g_background_scanning = false;
    g_userfault_handling = false;

    printf("\n==== CUDA-HOOK API TIMING SUMMARY ====\n");
    printf("%-32s %12s %12s %12s\n", "API", "calls", "total(ms)", "avg(us)");
    for (int i = 0; i < API_COUNT; ++i) {
        uint64_t calls = g_api_stats[i].calls.load(std::memory_order_relaxed);
        uint64_t total_ns = g_api_stats[i].total_ns.load(std::memory_order_relaxed);
        if (calls == 0) continue;
        double total_ms = double(total_ns) / 1e6;
        double avg_us = double(total_ns) / double(calls) / 1e3;
        printf("%-32s %12llu %12.3f %12.3f\n",
               g_api_names[i],
               (unsigned long long)calls,
               total_ms,
               avg_us);
    }
    printf("======================================\n");
}
