#include "libcudahook.h"
#include <sys/ioctl.h>

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

SharedPageInfo* find_page_by_hash(const char* hash, bool debug) {
    int i;
    for (i = 0; i < MAX_DEDUPE_PAGES; i++) {
        if (g_shared_memory->pages[i].valid && 
            strncmp(g_shared_memory->pages[i].hash, hash, HASH_SIZE) == 0) {
            return &g_shared_memory->pages[i];
        }
    }
    return nullptr;
}

void* background_scan_thread(void* arg) {
    const int initial_scan_interval_ms = g_hook_config.initial_scan_interval_ms; 
    const int max_scan_interval_ms = g_hook_config.max_scan_interval_ms;    
    const int stable_threshold = g_hook_config.stable_threshold;            
    
    int scan_interval_ms = initial_scan_interval_ms;
    int consecutive_no_dedup = 0;

    usleep(g_hook_config.boot_scan_interval_ms * 1000);
    while (g_background_scanning) {
        usleep(scan_interval_ms * 1000);
        ScopedTimer t_hook(API_BACKGROUND_SCAN);
        g_background_scans++;
        
        size_t pages_scanned = 0;
        pages_deduped = 0;
        size_t hash_calculated = 0;
        size_t stable_candidates = 0;
        
        printf("[%s] Starting background deduplication scan #%zu (interval: %dms)\n", 
               current_time().c_str(), g_background_scans, scan_interval_ms);
        
        //TODO scan持有锁时间太长
        std::lock_guard<std::mutex> lock(g_mutex);
        uint64_t start = now_ns();
        std::vector<CUdeviceptr> to_scan;
        {
            for (auto& page_item : *g_page_map) {
                if (page_item.second.is_deduped == false) {
                    to_scan.push_back(page_item.first);
                }
            }
        }
        
        pages_scanned = to_scan.size();
        
        for (CUdeviceptr va : to_scan) {
            LocalPageInfo& page_info = (*g_page_map)[va];
            std::string current_hash_str;
            {
                ScopedTimer t_hook(API_HASH_XXH64_GPU); 
                current_hash_str = gpu_fast_hash_single(va, g_page_size);
            }
            const char* current_hash = current_hash_str.c_str();
            hash_calculated++;

            // if (current_hash[0] == '\0' || strncmp(current_hash, "0000000000000000", 16) == 0)
            if (current_hash[0] == '\0')
                continue;
            

            if (page_info.hash[0] == '\0')
                strncpy(page_info.prev_hash, current_hash, HASH_SIZE);
            else 
                strncpy(page_info.prev_hash, page_info.hash, HASH_SIZE);
            strncpy(page_info.hash, current_hash, HASH_SIZE);
            //分区采样
            if (strncmp(page_info.prev_hash, page_info.hash, HASH_SIZE) == 0) {
                page_info.unchanged_scans++;
                if (page_info.unchanged_scans >= stable_threshold)
                    page_info.is_stable = true;
                if (page_info.is_stable) {
                    stable_candidates++;
                    CUmemAccessDesc accessDesc;
                    memset(&accessDesc, 0, sizeof(accessDesc));
                    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                    accessDesc.location.id = DEDUP_DEVICE;
                    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READ;
                    {
                        ScopedTimer t_hook(API_CU_MEM_SETACCESS_ORIG);
                        CUresult res = orig_cuMemSetAccess(va, g_page_size, &accessDesc, 1);
                        // TODO 加写保护太早，try_deduplicate_page还没开始导出就导致了page fault
                        // printf("[%s] GSH orig_cuMemSetAccess(va=%lx)\n", current_time().c_str(), va);
                        if (res != CUDA_SUCCESS) {
                            printf("cuMemSetAccess fail, res:%d\n", res);
                            return NULL;
                        }
                    }

                    if (try_deduplicate_page(va, NULL, g_page_size, current_hash)) {
                        page_info.is_deduped = true;
                        // printf("[%s] scan deduplicate page: va=%lx  current_hash=%s\n", current_time().c_str(), va, current_hash);
                    } else {
                        page_info.unchanged_scans = 0;
                        page_info.is_stable = false;
                        page_info.is_deduped = false;
                    }
                }
            } else {
                page_info.unchanged_scans = 0;
                page_info.is_stable = false;
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
        printf("[%s] Background scan completed. Page total: %d, Scanned: %zu, Hash calculated: %zu, "
               "Stable candidates: %zu, Deduped: %zu, New interval: %dms, total_ms: %.3f\n",
               current_time().c_str(), 
               g_page_map->size(),
               pages_scanned, 
               hash_calculated,
               stable_candidates,
               pages_deduped,
               scan_interval_ms,
               total_ms);
    }
    
    return NULL;
}

void* ipc_handler_thread(void* arg) {
    while (g_ipc_state.running) {
        int client = accept(g_ipc_state.ipc_sock, NULL, NULL);
        if (client < 0) {
            if (g_ipc_state.running) perror("accept failed");
            continue;
        }
        
        char request;
        if (read(client, &request, 1) != 1) {
            close(client);
            continue;
        }
        
        if (request == 'R') {
            char hash[HASH_SIZE];
            if (read(client, hash, HASH_SIZE) != HASH_SIZE) {
                close(client);
                continue;
            }
            
            // std::lock_guard<std::mutex> lock(g_mutex);
            CUmemGenericAllocationHandle handle = 0;
            for (const auto& kv : *g_page_map) {
                if (strncmp(kv.second.hash, hash, HASH_SIZE) == 0) {
                    CUdeviceptr va = kv.second.va;
                    handle = kv.second.handle;
                    break;
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
                } else {
                    fprintf(stderr, "Export failed in IPC handler: %d\n", res);
                }
            } else {
                printf("Hash not found in local pages: %s\n", hash);
            }
        } else if (request == 'T') { 
            struct IpcRequest req;
            if (read(client, &req, sizeof(req)) != sizeof(req)) {
                close(client);
                continue;
            }

            IpcResponse resp = {0};
            pthread_mutex_lock(&g_shared_memory->mutex);
            SharedPageInfo* page = find_page_by_hash(req.hash, false);
            if (page && page->owner_pid == req.src_pid) {
                auto page_it = g_page_map->find(page->owner_va);
                if (page_it != g_page_map->end()) {
                    LocalPageInfo& page = page_it->second;
                    resp.success = 1;
                    printf("Accepted ownership for page (Hash:%s)\n", req.hash);
                }
            }
            pthread_mutex_unlock(&g_shared_memory->mutex);
            write(client, &resp, sizeof(resp));
        } 
        close(client);
    }
    return NULL;
}

void init_ipc() {
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
    
    if (pthread_create(&g_ipc_state.handler_thread, NULL, ipc_handler_thread, NULL)) {
        perror("pthread_create failed");
        return;
    }
    
    g_ipc_state.running = true;
}

void* userfault_handler_thread(void* arg) {
    int fd = userfault_fd, ret;
    unsigned long long fault_addr;
    pid_t cur_pid;
    struct fault_info info;
    CUcontext cuda_ctx = (CUcontext)arg;
    size_t page_mask = ~(g_page_size - 1);

    CUresult res = cuCtxSetCurrent(cuda_ctx);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "cuCtxSetCurrent failed: %d\n", res);
    }
    
    while(g_userfault_handling) {
        info.fault_addr = 0;
        ret = read(fd, (void *)&info, 16);
        if (ret == -1) {
            printf("read error, errno:%d\n", errno);
            break;
        }
        if (ret != 16 || info.fault_addr == 0) continue;
        fault_addr = info.fault_addr;
        cur_pid = info.pid;
        CUdeviceptr base_addr = (CUdeviceptr)((uintptr_t)fault_addr & page_mask);
        SharedPageInfo* shared_page = nullptr;
        LocalPageInfo& fault_page = (g_page_map->find(base_addr))->second;

        if (info.fault_type != 3) 
            fprintf(stderr, "[%s] get fault_addr:0x%lx  base_addr:0x%lx  cur_pid:%d  fault_type:%d  fault_page.hash:%s\n", current_time().c_str(), fault_addr, base_addr, cur_pid, info.fault_type, fault_page.hash);
        //else 
        //    printf("[%s] get fault_addr:0x%lx  base_addr:0x%lx  cur_pid:%d  fault_type:%d  fault_page.hash:%s\n", current_time().c_str(), fault_addr, base_addr, cur_pid, info.fault_type, fault_page.hash);

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

        ScopedTimer t_hook(API_USERFAULT_HANDLE);
        pthread_mutex_lock(&g_shared_memory->mutex);
        shared_page = find_page_by_hash(fault_page.hash, true);

        if (shared_page != nullptr) {
            if (shared_page->ref_count > 0)
                shared_page->ref_count--;
            if (g_dedup_pid == shared_page->owner_pid && base_addr == shared_page->owner_va) { // owner页
                if (shared_page->ref_count == 0) { 
                    shared_page->valid = false;
                    g_shared_memory->page_count--;

                    CUmemAccessDesc accessDesc = {
                        {CU_MEM_LOCATION_TYPE_DEVICE, DEDUP_DEVICE},
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
                        g_shared_memory->page_count--;
                        return NULL;
                    }

                    if (new_owner_pid == g_dedup_pid) { // 转移物理handle的所有权就可以了
                        LocalPageInfo& new_owner_page = (g_page_map->find(new_owner_va))->second;
                        CUmemGenericAllocationHandle new_handle;
                        CUmemAllocationProp prop;
                        memset(&prop, 0, sizeof(prop));
                        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
                        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                        prop.location.id = DEDUP_DEVICE;
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
                                {CU_MEM_LOCATION_TYPE_DEVICE, DEDUP_DEVICE},
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
                            prop.location.id = DEDUP_DEVICE;
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
                                    {CU_MEM_LOCATION_TYPE_DEVICE, DEDUP_DEVICE},
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
                prop.location.id = DEDUP_DEVICE;
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
                accessDesc.location.id = DEDUP_DEVICE;
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
                    g_shared_memory->page_count--;
                }
            
                fault_page.handle = new_handle;
                fault_page.is_exported = false;
                g_saved_memory -= g_page_size;
            }
            // printf("[%s] CoW completed for fault_addr:0x%lx base_addr:0x%lx (Hash:%s)\n", current_time().c_str(), fault_addr, base_addr, shared_page->hash);
        } else {
            fprintf(stderr, "[%s] CoW shared page not found, fault_addr:0x%lx base_addr:0x%lx shared_page:%lx shared_page->hash:%s pid=%d\n", current_time().c_str(), fault_addr, base_addr, shared_page, shared_page->hash, g_dedup_pid);
            CUmemAccessDesc accessDesc;
            memset(&accessDesc, 0, sizeof(accessDesc));
            accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            accessDesc.location.id = DEDUP_DEVICE;
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
        pthread_mutex_unlock(&g_shared_memory->mutex);
    }
}

void init_userfault() {
    CUdevice cuDevice = -1;
    CUcontext ctx;
    orig_cuCtxGetCurrent(&ctx);
    cuCtxGetDevice(&cuDevice);
    userfault_fd = open(g_hook_config.userfault_fd_path, O_RDWR, O_CREAT);
    printf("userfault_fd %d\n",userfault_fd);
    CUuuid uuid;
    cuDeviceGetUuid(&uuid, cuDevice);
    int ret = ioctl(userfault_fd, CMD_START_USERFAULT, *(unsigned long*)&uuid);
    if (ret == -1) {
        printf("ioctl CMD_START_USERFAULT failed\n");
        close(userfault_fd);
        exit(-1);
    }
    
    g_userfault_handling = true;
    if (pthread_create(&userfault_handler, NULL, userfault_handler_thread, (void *)ctx)) {
        perror("pthread_create failed");
        return;
    }
}

void init_background_scan() {
    for (int i = 0; i < API_COUNT; ++i) {
        g_api_stats[i].calls.store(0, std::memory_order_relaxed);
        g_api_stats[i].total_ns.store(0, std::memory_order_relaxed);
    }
    // std::thread(monitor_thread_func).detach();
    // printf("Started statistics monitor thread\n");
    if (g_hook_config.enable_background_scan) {
        g_background_scanning = true;
        if (pthread_create(&background_scan, NULL, background_scan_thread, NULL)) {
            perror("pthread_create failed for background deduplication thread");
        } else {
            printf("Background deduplication thread started\n");
        }
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
        g_shared_memory->page_count = 0;
        for (int i = 0; i < MAX_DEDUPE_PAGES; i++) {
            g_shared_memory->pages[i].valid = false;
            memset(&g_shared_memory->pages[i], 0, sizeof(SharedPageInfo));
        }
        printf("Shared memory initialized by first process\n");
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
    ScopedTimer t_hook(API_TRY_DEDUPLICATE_PAGE); 

    // printf("[%s] try_deduplicate_page(va=0x%llx, size=%s, owner_va=%llx, hash=%s)\n", 
    //        current_time().c_str(), 
    //        (unsigned long long)va,
    //        format_bytes(size).c_str(),
    //        shared_page != NULL ? shared_page->owner_va : NULL, 
    //        hash);
    
    //////////////////////////////////////////////////// xxh64
    // std::string hash_str;
    // {
    //     ScopedTimer t_hook(API_HASH_XXH64); 
    //     hash_str = fast_hash(data, size);
    // }
    // const char* hash = hash_str.c_str(); 

    //////////////////////////////////////////////////// xxh64 gpu

    int ret = ioctl(userfault_fd, CMD_UPDATE_PAGEFAULT, *(unsigned long*)&va);
    if (ret == -1) {
        fprintf(stderr, "ioctl CMD_UPDATE_PAGEFAULT failed\n");
        exit(-1);
    }

    std::string hash_str;
    {
        ScopedTimer t_hook(API_HASH_XXH64_GPU); 
        hash_str = gpu_fast_hash_single(va, g_page_size);
    }
    const char* hash = hash_str.c_str();
    if (current_hash != NULL) {
        if (strcmp(current_hash, hash) != 0) 
            return false;
    }

    // CUmemAccessDesc accessDesc;
    // memset(&accessDesc, 0, sizeof(accessDesc));
    // accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    // accessDesc.location.id = DEDUP_DEVICE;
    // accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READ;
    // {
    //     ScopedTimer t_hook(API_CU_MEM_SETACCESS_ORIG); 
    //     if (orig_cuMemSetAccess(va, size, &accessDesc, 1) != CUDA_SUCCESS) {
    //         return false;
    //     }
    // }

    auto cur_page_it = g_page_map->find(va);
    if (cur_page_it == g_page_map->end()) {
        return false;
    }

    LocalPageInfo& cur_page = cur_page_it->second;
    cur_page.pagefault_flag = false;
    cur_page.is_stable = true;
    cur_page.memcpy_cnt++;
    if (cur_page.memcpy_cnt >= MAX_MEMCPY_LIMIT) 
        cur_page.is_stable = false;
    
    pthread_mutex_lock(&g_shared_memory->mutex);

    SharedPageInfo* shared_page = nullptr;
    
    //TODO: 一开始先在本地找，再在全局找
    shared_page = find_page_by_hash(hash, true);

    if (shared_page) {
        if (shared_page->owner_pid == g_dedup_pid && shared_page->owner_va == va) {
            // fprintf(stderr, "Error : shared_page->owner_pid=%d  shared_page->owner_va=%d\n", shared_page->owner_pid, shared_page->owner_va);
            return false;
        }
        CUmemGenericAllocationHandle imported_handle;
        if (g_dedup_pid == shared_page->owner_pid) {
            auto page_it = g_page_map->find(shared_page->owner_va);
            LocalPageInfo& page = page_it->second;
            imported_handle = page.handle;
        } else {
            int shareable_fd = ipc_get_shared_fd(
                shared_page->owner_pid,
                shared_page->sock_path,
                shared_page->hash
            );
            
            if (shareable_fd < 0) {
                pthread_mutex_unlock(&g_shared_memory->mutex);
                return false;
            }

            CUresult import_res;
            {
                ScopedTimer t_hook(API_CU_MEM_IMPORT_ORIG); 
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
                pthread_mutex_unlock(&g_shared_memory->mutex);
                return false;
            }
        }
        
        {
            ScopedTimer t_hook(API_CU_MEM_UNMAP_ORIG); 
            if (orig_cuMemUnmap(va, size) != CUDA_SUCCESS) {
                orig_cuMemRelease(imported_handle);
                pthread_mutex_unlock(&g_shared_memory->mutex);
                return false;
            }
        }

        {
            ScopedTimer t_hook(API_CU_MEM_MAP_ORIG); 
            if (orig_cuMemMap(va, size, 0, imported_handle, 0) != CUDA_SUCCESS) {
                orig_cuMemMap(va, size, 0, cur_page.handle, 0);
                orig_cuMemRelease(imported_handle);
                pthread_mutex_unlock(&g_shared_memory->mutex);
                return false;
            }
        }

        CUmemAccessDesc accessDesc;
        memset(&accessDesc, 0, sizeof(accessDesc));
        accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        accessDesc.location.id = DEDUP_DEVICE;
        accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READ;
        {
            ScopedTimer t_hook(API_CU_MEM_SETACCESS_ORIG); 
            if (orig_cuMemSetAccess(va, size, &accessDesc, 1) != CUDA_SUCCESS) {
                // 回滚操作
                orig_cuMemUnmap(va, size);
                orig_cuMemMap(va, size, 0, cur_page.handle, 0);
                orig_cuMemRelease(imported_handle);
                pthread_mutex_unlock(&g_shared_memory->mutex);
                return false;
            }
        }

        CUmemGenericAllocationHandle old_handle = cur_page.handle;
        cur_page.handle = imported_handle;
        cur_page.is_exported = true;
        strcpy(cur_page.hash,hash);

        {
            ScopedTimer t_hook(API_CU_MEM_RELEASE_ORIG); 
            orig_cuMemRelease(old_handle);
            g_allocations_create.erase(old_handle);
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
            printf("[%s] Dedup at VA: 0x%llx, Saved: %s, pid: %d, hash: %s\n",
                current_time().c_str(),
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

        pthread_mutex_unlock(&g_shared_memory->mutex);
        pages_deduped++;
        return true;
    } else {
        ShareableHandle shareable_handle;
        CUresult export_res;
        if (cur_page.is_exported == false) {
            ScopedTimer t_hook(API_CU_MEM_EXPORT_ORIG); 
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
                    if (!g_shared_memory->pages[i].valid) {
                        free_index = i;
                        break;
                    }
                }
                
                if (free_index == -1) {
                    fprintf(stderr, "Global dedupe page storage full!\n");
                    pthread_mutex_unlock(&g_shared_memory->mutex);
                    return false;
                }
                
                SharedPageInfo* new_shared_page = &g_shared_memory->pages[free_index];
                memset(new_shared_page, 0, sizeof(SharedPageInfo)); 
                
                strncpy(new_shared_page->hash, hash, HASH_SIZE - 1);
                new_shared_page->hash[HASH_SIZE - 1] = '\0';
                new_shared_page->owner_pid = g_dedup_pid;
                strncpy(new_shared_page->sock_path, 
                        ipc_get_current_sock_path().c_str(), 
                        sizeof(new_shared_page->sock_path)-1);
                new_shared_page->ref_count = 1;
                new_shared_page->valid = true;
                new_shared_page->device_id = DEDUP_DEVICE;
                new_shared_page->owner_va = va;
                g_shared_memory->page_count++;
                // 更新本地页信息
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
                if (!g_shared_memory->pages[i].valid) {
                    free_index = i;
                    break;
                }
            }
            
            if (free_index == -1) {
                fprintf(stderr, "Global dedupe page storage full!\n");
                pthread_mutex_unlock(&g_shared_memory->mutex);
                return false;
            }
            
            SharedPageInfo* new_shared_page = &g_shared_memory->pages[free_index];
            memset(new_shared_page, 0, sizeof(SharedPageInfo)); 
            
            strncpy(new_shared_page->hash, hash, HASH_SIZE - 1);
            new_shared_page->hash[HASH_SIZE - 1] = '\0';
            new_shared_page->owner_pid = g_dedup_pid;
            strncpy(new_shared_page->sock_path, 
                    ipc_get_current_sock_path().c_str(), 
                    sizeof(new_shared_page->sock_path)-1);
            new_shared_page->ref_count = 1;
            new_shared_page->valid = true;
            new_shared_page->device_id = DEDUP_DEVICE;
            new_shared_page->owner_va = va;
            g_shared_memory->page_count++;

            strncpy(cur_page.hash, hash, HASH_SIZE);
        }
    }
    pthread_mutex_unlock(&g_shared_memory->mutex);
    return true;
}

extern "C" {
CUresult CUDAAPI cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytesize) {
    ScopedTimer t_hook(API_CU_MEM_ALLOC_HOOK); 
    if (userfault_fd == -1) {
        printf("Initializing shared memory...\n");
        init_shared_memory();
        printf("Shared memory initialized at %p\n", (void*)g_shared_memory);
        
        printf("Initializing userfault...\n");
        init_userfault();
        printf("userfault initialized\n");

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
            printf("[%s] cuMemAlloc_v2(dptr=0x%llx, size=%s)\n", 
                current_time().c_str(), 
                (unsigned long long)*dptr,
                format_bytes(bytesize).c_str());
        }
        return res;
    }
    CUcontext ctx;    
    
    CUmemAllocationProp prop;
    memset(&prop, 0, sizeof(prop));
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = DEDUP_DEVICE;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    
    size_t aligned_size = ((bytesize + g_page_size - 1) / g_page_size) * g_page_size;
    size_t num_pages = aligned_size / g_page_size;
    VMMAllocInfo alloc_info;
    alloc_info.aligned_size = aligned_size;
    
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
            cuMemRelease(handle);
            return res;
        }
        LocalPageInfo page;
        page.handle = handle;
        page.va = va_ptr;
        memset(page.hash, 0, HASH_SIZE);
        alloc_info.pages.push_back(page);
        (*g_page_map)[va_ptr] = page;
    }

    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = DEDUP_DEVICE;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    
    {
    ScopedTimer t_orig_setaccess(API_CU_MEM_SETACCESS_ORIG);
    res = orig_cuMemSetAccess(va, aligned_size, &accessDesc, 1);
    }
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "cuMemSetAccess failed: %d\n", res);
        cuMemUnmap(va, aligned_size);
        return res;
    }

    alloc_info.va = va;
    g_allocations_vmm[va] = alloc_info;
    
    *dptr = va;
    printf("[%s] cuMemAlloc_v2(dptr=0x%llx, aligned_size=%s, g_total_allocated=%s)\n", 
        current_time().c_str(), 
        (unsigned long long)*dptr,
        format_bytes(aligned_size).c_str(),
        format_bytes(g_total_allocated).c_str());

    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemAlloc(CUdeviceptr* dptr, size_t bytesize) {
    ScopedTimer t_hook(API_CU_MEM_ALLOC_HOOK); 
    if (userfault_fd == -1) {
        printf("Initializing shared memory...\n");
        init_shared_memory();
        printf("Shared memory initialized at %p\n", (void*)g_shared_memory);
        
        printf("Initializing userfault...\n");
        init_userfault();
        printf("userfault initialized\n");

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
            printf("[%s] cuMemAlloc(dptr=0x%llx, size=%s)\n", 
                current_time().c_str(), 
                (unsigned long long)*dptr,
                format_bytes(bytesize).c_str());
        }
        return res;
    }
    CUcontext ctx;    
    
    CUmemAllocationProp prop;
    memset(&prop, 0, sizeof(prop));
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = DEDUP_DEVICE;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    
    size_t aligned_size = ((bytesize + g_page_size - 1) / g_page_size) * g_page_size;
    size_t num_pages = aligned_size / g_page_size;
    VMMAllocInfo alloc_info;
    alloc_info.aligned_size = aligned_size;
    
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
            cuMemRelease(handle);
            return res;
        }
        LocalPageInfo page;
        page.handle = handle;
        page.va = va_ptr;
        memset(page.hash, 0, HASH_SIZE);
        alloc_info.pages.push_back(page);
        (*g_page_map)[va_ptr] = page;
    }

    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = DEDUP_DEVICE;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    
    {
    ScopedTimer t_orig_setaccess(API_CU_MEM_SETACCESS_ORIG);
    res = orig_cuMemSetAccess(va, aligned_size, &accessDesc, 1);
    }
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "cuMemSetAccess failed: %d\n", res);
        cuMemUnmap(va, aligned_size);
        return res;
    }

    alloc_info.va = va;
    g_allocations_vmm[va] = alloc_info;
    
    *dptr = va;
    // printf("[%s] cuMemAlloc(dptr=0x%llx, aligned_size=%s, g_total_allocated=%s)\n", 
    //     current_time().c_str(), 
    //     (unsigned long long)*dptr,
    //     format_bytes(aligned_size).c_str(),
    //     format_bytes(g_total_allocated).c_str());

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
            printf("hook in cuMemcpyHtoD_v2, res=%d dstDevice=%lx\n", res, dstDevice);
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
        
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            auto page_it = g_page_map->lower_bound(current_addr);
            if (page_it != g_page_map->end() && current_addr == page_it->first){
                page_offset = 0;
                page_bytes = std::min((int)g_page_size, bytes_remaining);
                page_start = page_it->first;
                in_vmm = true;
            }
            if (page_it != g_page_map->begin()) {
                auto page_it0 = page_it;
                page_it--;
                if (current_addr > page_it->first && current_addr < page_it->first + g_page_size){
                    page_offset = current_addr - page_it->first;
                    page_bytes = g_page_size - page_offset;
                    page_start = page_it->first + g_page_size;
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
                printf("hook in cuMemcpyHtoD_v2, res=%d current_addr=%lx\n", res, current_addr);
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
            printf("hook in cuMemcpyHtoD_v2, res=%d current_addr=%lx\n", res, current_addr);
            if (res == 1) {
                CUmemAccessDesc accessDesc;
                memset(&accessDesc, 0, sizeof(accessDesc));
                accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                accessDesc.location.id = DEDUP_DEVICE;
                accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                {
                    ScopedTimer t_hook(API_CU_MEM_SETACCESS_ORIG);
                    res = orig_cuMemSetAccess(current_addr, g_page_size, &accessDesc, 1);
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
                    printf("hook in cuMemcpyHtoD_v2, res=%d current_addr=%lx\n", res, current_addr);
            } else return res;
        }
        
        if (page_offset == 0 && page_bytes == g_page_size) {
            // 对于不稳定页，这里就不加写保护了，也不进入try_deduplicate_page，而是通过scan来处理
            LocalPageInfo& page = (g_page_map->find(page_start))->second;
            if (page.memcpy_cnt < MAX_MEMCPY_LIMIT) {
                res = cuCtxSynchronize();
                if (res != CUDA_SUCCESS)
                    printf("cuCtxSynchronize fail\n");
                CUmemAccessDesc accessDesc;
                memset(&accessDesc, 0, sizeof(accessDesc));
                accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                accessDesc.location.id = DEDUP_DEVICE;
                accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READ;
                {
                    ScopedTimer t_hook(API_CU_MEM_SETACCESS_ORIG);
                    res = orig_cuMemSetAccess(page_start, g_page_size, &accessDesc, 1);
                }
                if (res != CUDA_SUCCESS) {
                    printf("cuMemSetAccess fail, res:%d\n", res);
                    return res;
                }

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
            printf("hook in cuMemcpyHtoD_v2, res=%d dstDevice=%lx\n", res, dstDevice);
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
        
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            auto page_it = g_page_map->lower_bound(current_addr);
            if (page_it != g_page_map->end() && current_addr == page_it->first){
                page_offset = 0;
                page_bytes = std::min((int)g_page_size, bytes_remaining);
                page_start = page_it->first;
                in_vmm = true;
            }
            if (page_it != g_page_map->begin()) {
                auto page_it0 = page_it;
                page_it--;
                if (current_addr > page_it->first && current_addr < page_it->first + g_page_size){
                    page_offset = current_addr - page_it->first;
                    page_bytes = g_page_size - page_offset;
                    page_start = page_it->first + g_page_size;
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
                printf("hook in cuMemcpyHtoD_v2, res=%d current_addr=%lx\n", res, current_addr);
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
            printf("hook in cuMemcpyHtoD_v2, res=%d current_addr=%lx\n", res, current_addr);
            if (res == 1) {
                CUmemAccessDesc accessDesc;
                memset(&accessDesc, 0, sizeof(accessDesc));
                accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                accessDesc.location.id = DEDUP_DEVICE;
                accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                {
                    ScopedTimer t_hook(API_CU_MEM_SETACCESS_ORIG);
                    res = orig_cuMemSetAccess(current_addr, g_page_size, &accessDesc, 1);
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
                    printf("hook in cuMemcpyHtoD_v2, res=%d current_addr=%lx\n", res, current_addr);
            } else return res;
        }
        
        if (page_offset == 0 && page_bytes == g_page_size) {
            // 对于不稳定页，这里就不加写保护了，也不进入try_deduplicate_page，而是通过scan来处理
            LocalPageInfo& page = (g_page_map->find(page_start))->second;
            if (page.memcpy_cnt < MAX_MEMCPY_LIMIT) {
                res = cuCtxSynchronize();
                if (res != CUDA_SUCCESS)
                    printf("cuCtxSynchronize fail\n");
                CUmemAccessDesc accessDesc;
                memset(&accessDesc, 0, sizeof(accessDesc));
                accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                accessDesc.location.id = DEDUP_DEVICE;
                accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READ;
                {
                    ScopedTimer t_hook(API_CU_MEM_SETACCESS_ORIG);
                    res = orig_cuMemSetAccess(page_start, g_page_size, &accessDesc, 1);
                }
                if (res != CUDA_SUCCESS) {
                    printf("cuMemSetAccess fail, res:%d\n", res);
                    return res;
                }

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
            printf("hook in cuMemcpyHtoD_v2, res=%d dstDevice=%lx\n", res, dstDevice);
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
        
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            auto page_it = g_page_map->lower_bound(current_addr);
            if (page_it != g_page_map->end() && current_addr == page_it->first){
                page_offset = 0;
                page_bytes = std::min((int)g_page_size, bytes_remaining);
                page_start = page_it->first;
                in_vmm = true;
            }
            if (page_it != g_page_map->begin()) {
                auto page_it0 = page_it;
                page_it--;
                if (current_addr > page_it->first && current_addr < page_it->first + g_page_size){
                    page_offset = current_addr - page_it->first;
                    page_bytes = g_page_size - page_offset;
                    page_start = page_it->first + g_page_size;
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
                printf("hook in cuMemcpyHtoD_v2, res=%d current_addr=%lx\n", res, current_addr);
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
            printf("hook in cuMemcpyHtoD_v2, res=%d current_addr=%lx\n", res, current_addr);
            if (res == 1) {
                CUmemAccessDesc accessDesc;
                memset(&accessDesc, 0, sizeof(accessDesc));
                accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                accessDesc.location.id = DEDUP_DEVICE;
                accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                {
                    ScopedTimer t_hook(API_CU_MEM_SETACCESS_ORIG);
                    res = orig_cuMemSetAccess(current_addr, g_page_size, &accessDesc, 1);
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
                    printf("hook in cuMemcpyHtoD_v2, res=%d current_addr=%lx\n", res, current_addr);
            } else return res;
        }
        
        if (page_offset == 0 && page_bytes == g_page_size) {
            LocalPageInfo& page = (g_page_map->find(page_start))->second;
            if (page.memcpy_cnt < MAX_MEMCPY_LIMIT) {
                {
                    ScopedTimer t_hook(API_CU_CTX_SYNCHRONIZE); 
                    res = cuCtxSynchronize();
                    if (res != CUDA_SUCCESS)
                        printf("cuCtxSynchronize fail\n");
                }
                CUmemAccessDesc accessDesc;
                memset(&accessDesc, 0, sizeof(accessDesc));
                accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                accessDesc.location.id = DEDUP_DEVICE;
                accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READ;
                {
                    ScopedTimer t_hook(API_CU_MEM_SETACCESS_ORIG);
                    res = orig_cuMemSetAccess(page_start, g_page_size, &accessDesc, 1);
                }
                if (res != CUDA_SUCCESS) {
                    printf("cuMemSetAccess fail, res:%d\n", res);
                    return res;
                }

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

CUresult cuMemFree_v2(CUdeviceptr dptr) {
    CUresult res = CUDA_ERROR_UNKNOWN;
    // printf("[%s] cuMemFree_v2(dptr=0x%llx)\n", 
    //        current_time().c_str(), 
    //        (unsigned long long)dptr);
    // 查找分配信息
    auto it = g_allocations_vmm.find(dptr);
    if (it == g_allocations_vmm.end()) {
        return orig_cuMemFree(dptr);
    }
    VMMAllocInfo& alloc_info = it->second;
    int num_pages = alloc_info.pages.size();
    for (size_t i = 0; i < num_pages; i++) {
        LocalPageInfo p = alloc_info.pages[i];
        CUdeviceptr va_ptr = alloc_info.va + i * g_page_size;
        CUresult res1 = cuMemUnmap(va_ptr, g_page_size);
        CUresult res2 = cuMemRelease(p.handle);
        if (res1 != CUDA_SUCCESS || res2 != CUDA_SUCCESS) {
            fprintf(stderr, "cuMemFree: page free error: 0x%llx\n", (unsigned long long)va_ptr);
            return CUDA_ERROR_INVALID_VALUE;
        }
    } 
    res = cuMemAddressFree(alloc_info.va, alloc_info.aligned_size);
    // 从全局映射中移除
    g_allocations_vmm.erase(it);
    return res;
}

CUresult cuMemFree(CUdeviceptr dptr) {
    CUresult res = CUDA_ERROR_UNKNOWN;
    // printf("[%s] cuMemFree(dptr=0x%llx)\n", 
    //        current_time().c_str(), 
    //        (unsigned long long)dptr);
    // 查找分配信息
    auto it = g_allocations_vmm.find(dptr);
    if (it == g_allocations_vmm.end()) return CUDA_SUCCESS;
    VMMAllocInfo& alloc_info = it->second;
    int num_pages = alloc_info.pages.size();
    for (size_t i = 0; i < num_pages; i++) {
        LocalPageInfo p = alloc_info.pages[i];
        CUdeviceptr va_ptr = alloc_info.va + i * g_page_size;
        CUresult res1 = cuMemUnmap(va_ptr, g_page_size);
        CUresult res2 = cuMemRelease(p.handle);
        if (res1 != CUDA_SUCCESS || res2 != CUDA_SUCCESS) {
            fprintf(stderr, "cuMemFree: page free error: 0x%llx\n", (unsigned long long)va_ptr);
            return CUDA_ERROR_INVALID_VALUE;
        }
    } 
    res = cuMemAddressFree(alloc_info.va, alloc_info.aligned_size);
    // 从全局映射中移除
    g_allocations_vmm.erase(it);
    return res;
}

CUresult CUDAAPI cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream) {
    CUresult res = CUDA_ERROR_UNKNOWN;
    // printf("[%s] cuMemFreeAsync(dptr=0x%llx)\n", 
    //        current_time().c_str(), 
    //        (unsigned long long)dptr);
    auto it = g_allocations_vmm.find(dptr);
    if (it == g_allocations_vmm.end()) return CUDA_SUCCESS;
    VMMAllocInfo& alloc_info = it->second;
    int num_pages = alloc_info.pages.size();
    for (size_t i = 0; i < num_pages; i++) {
        LocalPageInfo p = alloc_info.pages[i];
        CUdeviceptr va_ptr = alloc_info.va + i * g_page_size;
        CUresult res1 = cuMemUnmap(va_ptr, g_page_size);
        CUresult res2 = cuMemRelease(p.handle);
        if (res1 != CUDA_SUCCESS || res2 != CUDA_SUCCESS) {
            fprintf(stderr, "cuMemFreeAsync: page free error: 0x%llx\n", (unsigned long long)va_ptr);
            return CUDA_ERROR_INVALID_VALUE;
        }
    } 
    res = cuMemAddressFree(alloc_info.va, alloc_info.aligned_size);
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
        g_total_allocated += aligned_size;
        g_current_allocated += aligned_size;
        g_allocations_create[*handle] = size;
    }
    
    return res;
}

// cuMemMap hook
CUresult CUDAAPI cuMemMap(CUdeviceptr va, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags) {
    CUresult res = orig_cuMemMap ? 
                   orig_cuMemMap(va, size, offset, handle, flags) : 
                   CUDA_ERROR_NOT_INITIALIZED;
    
    if (res == CUDA_SUCCESS && size == g_page_size) {
        printf("[%s] cuMemMap(va=0x%llx, size=%s)\n", 
               current_time().c_str(), 
               (unsigned long long)va,
               format_bytes(size).c_str());
    }
    return res;
}

// // cuMemSetAccess hook
CUresult CUDAAPI cuMemSetAccess(CUdeviceptr va, size_t size, const CUmemAccessDesc* desc, size_t count) {
    CUresult res = orig_cuMemSetAccess ? 
                   orig_cuMemSetAccess(va, size, desc, count) : 
                   CUDA_ERROR_NOT_INITIALIZED;
    
    if (res == CUDA_SUCCESS) {
        printf("[%s] cuMemSetAccess(va=0x%llx, size=%s, count=%zu)\n", 
               current_time().c_str(), 
               (unsigned long long)va,
               format_bytes(size).c_str(),
               count);
    }
    
    return res;
}

// cuMemUnmap hook
CUresult CUDAAPI cuMemUnmap(CUdeviceptr va, size_t size) {
    CUresult res = orig_cuMemUnmap ? 
                   orig_cuMemUnmap(va, size) : 
                   CUDA_ERROR_NOT_INITIALIZED;
    
    if (res == CUDA_SUCCESS) {        
        if (size == g_page_size)
            g_page_map->erase(va);
    }
    
    return res;
}

// cuMemRelease hook
CUresult CUDAAPI cuMemRelease(CUmemGenericAllocationHandle handle) {
    auto it = g_allocations_create.find(handle);
    // 防止重复释放
    if (it == g_allocations_create.end()) {
        return CUDA_SUCCESS;
    } 
    CUresult res = orig_cuMemRelease ? 
                   orig_cuMemRelease(handle) : 
                   CUDA_ERROR_NOT_INITIALIZED;
                   
    g_current_allocated -= it->second;
    g_allocations_create.erase(it);
    return res;
}

// cuMemAddressReserve hook
CUresult CUDAAPI cuMemAddressReserve(CUdeviceptr* ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags) {
    CUresult res = orig_cuMemAddressReserve ? 
                   orig_cuMemAddressReserve(ptr, size, alignment, addr, flags) : 
                   CUDA_ERROR_NOT_INITIALIZED;
    
    if (res == CUDA_SUCCESS) {
        printf("[%s] cuMemAddressReserve(ptr=0x%llx, size=%s, alignment=%zu, addr=0x%llx, flags=0x%llx)\n", 
               current_time().c_str(), 
               (unsigned long long)*ptr,
               format_bytes(size).c_str(),
               alignment,
               (unsigned long long)addr,
               flags);
    }
    
    return res;
}

// cuMemAddressFree hook
CUresult CUDAAPI cuMemAddressFree(CUdeviceptr ptr, size_t size) {
    CUresult res = orig_cuMemAddressFree ? 
                   orig_cuMemAddressFree(ptr, size) : 
                   CUDA_ERROR_NOT_INITIALIZED;
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

