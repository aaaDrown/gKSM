#include "libcudahook.h"
#include <sys/ioctl.h>

//LD_LIBRARY_PATH=/home/taig/gsh/Deduplication/build/ ./a
//LD_LIBRARY_PATH=/home/taig/gsh/Deduplication/build/ ./matrix
//LD_LIBRARY_PATH=/home/taig/gsh/Deduplication/build/ python resnet-50.py 

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

    char proc_name[16]; 

    if (prctl(PR_GET_NAME, (unsigned long)proc_name) == 0) {
        printf("Process name: %s\n", proc_name);
    } else {
        perror("prctl");
    }

    get_command_line();

    // libcuda.so和libcuda.so.1只加载一个就行了
    if(!strcmp(lib_name,"./build/libcuda.so")) return;

    g_page_map = new std::map<CUdeviceptr, LocalPageInfo>();
    g_dedup_pid = getpid();
    
    printf("Initializing CUDA Driver API Hook...\n");
    init_hook_api();
    printf("CUDA Driver API Hook initialized, Page Size: %s, pid: %d, ppid:%d\n", format_bytes(g_page_size).c_str(), g_dedup_pid, getppid());
    
    printf("Initializing ipc...\n");
    init_ipc();
    printf("ipc initialized\n");
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
            // handle=0;
            if (handle) {
                // 导出为可共享的FD
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
        } 
        // else if (request == 'T') { 
        //     struct IpcRequest req;
        //     if (read(client, &req, sizeof(req)) != sizeof(req)) {
        //         close(client);
        //         continue;
        //     }
        //     IpcResponse resp = {0};
        //     pthread_mutex_lock(&g_shared_memory->mutex);
        //     SharedPageInfo* page = find_page_by_hash(req.hash);
        //     if (page && page->owner_pid == req.src_pid) {
        //         // 接受所有权转移
        //         page->owner_pid = getpid();
        //         // 更新本地映射
        //         auto page_it = g_page_map->find(page->owner_va);
        //         if (page_it != g_page_map->end()) {
        //             // 导入原始共享句柄
        //             CUresult import_res = orig_cuMemImportFromShareableHandle(
        //                 &page_it->second.handle,
        //                 req.shareable_fd,
        //                 CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        //             );
        //             if (import_res == CUDA_SUCCESS) {
        //                 // 重新映射到相同VA
        //                 orig_cuMemUnmap(page->owner_va, g_page_size);
        //                 orig_cuMemMap(page->owner_va, g_page_size, 0, 
        //                              page_it->second.handle, 0);
        //                 // 设置默认权限
        //                 CUmemAccessDesc accessDesc = {
        //                     {CU_MEM_LOCATION_TYPE_DEVICE, device},
        //                     CU_MEM_ACCESS_FLAGS_PROT_READ
        //                 };
        //                 orig_cuMemSetAccess(page->owner_va, g_page_size, &accessDesc, 1);
        //                 resp.success = 1;
        //                 printf("Accepted ownership for page (Hash:%s)\n", req.hash);
        //             }
        //         }
        //     }
        //     pthread_mutex_unlock(&g_shared_memory->mutex);
        //     write(client, &resp, sizeof(resp));
        // }
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
    // printf("IPC module initialized (PID: %d)\n", getpid());
}

void* userfault_handler_thread(void* arg) {
    int fd = userfault_fd, ret;
    unsigned long long fault_addr;
    pid_t cur_pid;
    struct fault_info info;
    CUcontext cuda_ctx = (CUcontext)arg;
    size_t page_mask = ~(g_page_size - 1);
    CUdevice device;

    cuCtxSetCurrent(cuda_ctx);
    CUresult res = cuCtxGetDevice(&device);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "cuCtxGetDevice failed: %d\n", res);
    }
    
    while(1) {
        ScopedTimer t_hook(API_USERFAULT_HANDLE);
        info.fault_addr = 0;
        // 保证读取的是当前进程的pagefault
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
        if (fault_page.dedup_flag == true) {
            ret = write(fd, "done", 4);
            if (ret == -1) {
                printf("write error, errno:%d\n", errno);
                break;
            }
            continue;
        }
        fault_page.dedup_flag = true;
        // printf("get fault_addr:0x%lx  base_addr:0x%lx  cur_pid:%d  fault_type:%d\n", fault_addr, base_addr, cur_pid, info.fault_type);

        // pthread_mutex_lock(&g_shared_memory->mutex);
        //TODO: 触发写之后页的内容无法再跟踪，COW之后的local_page.hash值为空
        for (int i = 0; i < MAX_DEDUPE_PAGES; i++) {
            if (g_shared_memory->pages[i].valid && 
                strncmp(g_shared_memory->pages[i].hash, fault_page.hash, HASH_SIZE) == 0) {
                shared_page = &g_shared_memory->pages[i];
                break;
            }
        }
        if (shared_page) {
            if (shared_page->ref_count > 0)
                shared_page->ref_count--;
            if (g_dedup_pid == shared_page->owner_pid && base_addr == shared_page->owner_va) { //当前进程owner页
                if (shared_page->ref_count == 0) {
                    // printf("Releasing shared owner page (Hash:%s)\n", shared_page->hash);
                    shared_page->valid = false;
                    g_shared_memory->page_count--;

                    CUmemAccessDesc accessDesc = {
                        {CU_MEM_LOCATION_TYPE_DEVICE, device},
                        CU_MEM_ACCESS_FLAGS_PROT_READWRITE
                    };
                    if (orig_cuMemSetAccess(base_addr, g_page_size, &accessDesc, 1) != CUDA_SUCCESS) {
                        printf("Failed to set access for new page\n");
                        break;
                    }
                } else {
                    // printf("switch owner page (Hash:%s)\n", shared_page->hash);
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
                            // printf("Selected new owner: PID=%d, VA=0x%lx\n", new_owner_pid, new_owner_va);
                            break;
                        }
                    }

                    if (!found_new_owner) {
                        for (int i = 0; i < MAX_SHARED_REFS; i++) { 
                            if (shared_page->ref_pids[i] != 0) {
                                new_owner_pid = shared_page->ref_pids[i];
                                new_owner_va = shared_page->ref_vas[i];
                                found_new_owner = true;
                                // printf("Selected new owner: PID=%d, VA=0x%lx\n", new_owner_pid, new_owner_va);
                                break;
                            }
                        }
                    }
                    
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
                        prop.location.id = device;
                        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

                        if (orig_cuMemCreate(&new_handle, g_page_size, &prop, 0) == CUDA_SUCCESS) {
                            if(orig_cuMemUnmap(base_addr, g_page_size) != CUDA_SUCCESS) {
                                printf("gsh in orig_cuMemUnmap error\n");
                                break;
                            }

                            if(orig_cuMemMap(base_addr, g_page_size, 0, new_handle, 0) != CUDA_SUCCESS) {
                                printf("gsh in orig_cuMemMap error\n");
                                break;
                            }

                            CUmemAccessDesc accessDesc = {
                                {CU_MEM_LOCATION_TYPE_DEVICE, device},
                                CU_MEM_ACCESS_FLAGS_PROT_READWRITE
                            };
                            orig_cuMemSetAccess(base_addr, g_page_size, &accessDesc, 1);
                            // TODO:这里如果利用页内数据对页内进行写入需要先复制再写
                            // CUcontext ctx;
                            // orig_cuCtxGetCurrent(&ctx);
                            // printf("cuCtxGetCurrent :%p\n", ctx);
                            // cuCtxSetCurrent(cuda_ctx);
                            // CUresult res = orig_cuMemcpyDtoD(base_addr, new_owner_va, g_page_size);
                            // if(res != CUDA_SUCCESS){
                            //     printf("switch owner error in orig_cuMemcpyDtoD : %d\n", res);
                            // } else {
                            //     printf("switch owner success in orig_cuMemcpyDtoD : %d\n", res);
                            // }
                        }

                        shared_page->owner_pid = g_dedup_pid;
                        shared_page->owner_va = new_owner_va;
                        new_owner_page.handle = fault_page.handle;
                        fault_page.handle = new_handle;
                        // printf("old owner va=%lx  new_handle=%lx\n", base_addr, new_handle);

                        g_saved_memory -= g_page_size;
                        printf("g_saved_memory=%s\n", format_bytes(g_saved_memory).c_str());

                    } else { 
                        // 在其他进程中寻找owner
                        // if (new_owner_pid != 0) {
                        //     // 创建IPC请求消息
                        //     struct IpcRequest req = {
                        //         .type = OWNERSHIP_TRANSFER,
                        //         .src_pid = g_dedup_pid,
                        //         .dst_pid = new_owner_pid,
                        //         .hash = {0}
                        //     };
                        //     memcpy(req.hash, shared_page->hash, HASH_SIZE);
                        //     // 通过Unix socket发送请求
                        //     int sock = connect_to_process(new_owner_pid);
                        //     if (sock >= 0) {
                        //         write(sock, &req, sizeof(req));
                        //         // 等待确认
                        //         IpcResponse resp;
                        //         if (read(sock, &resp, sizeof(resp)) == sizeof(resp) && resp.success) {
                        //             printf("Ownership transferred to PID=%d\n", new_owner_pid); 
                        //             // 更新共享页元数据
                        //             shared_page->owner_pid = new_owner_pid;
                        //             shared_page->owner_va = new_owner_va;
                        //         } else {
                        //             printf("Failed to transfer ownership to PID=%d\n", new_owner_pid);
                        //         }
                        //         close(sock);
                        //     } else {
                        //         printf("IPC connection failed to PID=%d\n", new_owner_pid);
                        //     }
                        // }
                    }                    
                }
            } else {// 当前进程非owner页
                if (g_dedup_pid != shared_page->owner_pid || shared_page->ref_count <= 0) {
                    printf("userfault handle fail g_dedup_pid=%d  shared_page->owner_pid=%d  shared_page->ref_count=%d\n", g_dedup_pid, shared_page->owner_pid, shared_page->ref_count);
                    ret = write(fd, "done", 4);
                    break;
                }
                CUmemAllocationProp prop = {};
                prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
                prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                prop.location.id = device;
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

                CUmemAccessDesc accessDesc = {
                    {CU_MEM_LOCATION_TYPE_DEVICE, device},
                    CU_MEM_ACCESS_FLAGS_PROT_READWRITE
                };
                if (orig_cuMemSetAccess(base_addr, g_page_size, &accessDesc, 1) != CUDA_SUCCESS) {
                    printf("Failed to set access for new page\n");
                    orig_cuMemUnmap(base_addr, g_page_size);
                    orig_cuMemMap(base_addr, g_page_size, 0, fault_page.handle, 0);
                    orig_cuMemRelease(new_handle);
                    continue;
                }

                CUresult res = orig_cuMemcpyDtoD(base_addr, shared_page->owner_va, g_page_size);
                if(res != CUDA_SUCCESS){
                    printf("gsh error in orig_cuMemcpyDtoD : %d\n", res);
                } 
                
                int i;
                for(i=0;i<MAX_SHARED_REFS;i++){
                    if(shared_page->ref_vas[i] == base_addr && shared_page->ref_pids[i] == g_dedup_pid){
                        shared_page->ref_vas[i] = 0;
                        shared_page->ref_pids[i] = 0;
                        break;
                    }
                }
                if(i == MAX_SHARED_REFS) {
                    printf("userfault handle fail\n");
                    break;
                }

                if (shared_page->ref_count == 0) {
                    printf("Releasing shared page (Hash:%s)\n", shared_page->hash);
                    shared_page->valid = false;
                    g_shared_memory->page_count--;
                }
            
                fault_page.handle = new_handle;
                g_saved_memory -= g_page_size;
                printf("g_saved_memory=%s\n", format_bytes(g_saved_memory).c_str());
                // memset(fault_page.hash, 0, HASH_SIZE); // 不再是共享页
            }
            // printf("CoW completed for fault_addr:0x%lx base_addr:0x%lx (Hash:%s)\n", fault_addr, base_addr, shared_page->hash);
        } else 
            printf("CoW shared page not found, fault_addr:0x%lx base_addr:0x%lx (Hash:%s)\n", fault_addr, base_addr, shared_page->hash);
        ret = write(fd, "done", 4);
        if (ret == -1) {
            printf("write error, errno:%d\n", errno);
            break;
        }
        // pthread_mutex_unlock(&g_shared_memory->mutex);
    }
}

void init_userfault() {
    CUdevice cuDevice = -1;
    CUcontext ctx;
    orig_cuCtxGetCurrent(&ctx);
    printf("cuCtxGetCurrent :%p\n", ctx);
    cuCtxGetDevice(&cuDevice);
    userfault_fd = open("/dev/dedup_dev", O_RDWR, O_CREAT);
    printf("userfault_fd %d\n",userfault_fd);
    CUuuid uuid;
    cuDeviceGetUuid(&uuid, cuDevice);
    int ret = ioctl(userfault_fd, CMD_START_USERFAULT, *(unsigned long*)&uuid);
    if (ret == -1) {
        printf("ioctl CMD_START_USERFAULT failed\n");
        close(userfault_fd);
        exit(-1);
    }
    pthread_t tid;
    if (pthread_create(&tid, NULL, userfault_handler_thread, (void *)ctx)) {
        perror("pthread_create failed");
        return;
    }
}

void init_shared_memory() {
    const char* SHM_NAME = "/cuda_dedup_shared";
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
    // 使用进程间原子操作来保护互斥锁的初始化
    static volatile int* mutex_initialized = &g_shared_memory->mutex_initialized;
    // 尝试原子性地设置互斥锁初始化标志
    if (__sync_val_compare_and_swap(mutex_initialized, 0, 1) == 0) {
        // 我们是第一个进程，需要初始化互斥锁
        pthread_mutexattr_t attr;
        pthread_mutexattr_init(&attr);
        pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
        pthread_mutex_init(&g_shared_memory->mutex, &attr);
        pthread_mutexattr_destroy(&attr);
        // 标记互斥锁已初始化完成
        __sync_add_and_fetch(mutex_initialized, 1);
    } else {
        // 等待其他进程完成互斥锁初始化
        while (*mutex_initialized != 2) {
            usleep(1000); // 短暂休眠
        }
    }
    // 现在可以安全地使用互斥锁了
    pthread_mutex_lock(&g_shared_memory->mutex);
    bool first_process = false;
    if (g_shared_memory->initialized != 0x12345678) {
        g_shared_memory->initialized = 0x12345678;
        g_shared_memory->page_count = 0;
        for (int i = 0; i < MAX_DEDUPE_PAGES; i++) {
            g_shared_memory->pages[i].valid = false;
            memset(&g_shared_memory->pages[i], 0, sizeof(SharedPageInfo));
        }
        first_process = true;
        printf("Shared memory initialized by first process\n");
    }
    printf("g_shared_memory->initialized=%d, g_shared_memory->page_count=%d\n",g_shared_memory->initialized, g_shared_memory->page_count);
    pthread_mutex_unlock(&g_shared_memory->mutex);
    if (!first_process) {
        printf("Attached to existing shared memory\n");
    }
}

void init_hook_api() {
    void* handle = dlopen(orig_path, RTLD_NOW | RTLD_NODELETE);
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

bool try_deduplicate_page(CUdeviceptr va, size_t size) {
    ScopedTimer t_hook(API_TRY_DEDUPLICATE_PAGE); 

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
        ScopedTimer t_hook(API_HASH_XXH64_GPU); 
        hash_str = gpu_fast_hash_single(va, g_page_size);
    }
    const char* hash = hash_str.c_str();

    // printf("hash=%s \n",hash);
    //TODO 先做写保护，再加hash
    auto cur_page_it = g_page_map->find(va);
    if (cur_page_it == g_page_map->end()) {
        return false;
    }

    LocalPageInfo& cur_page = cur_page_it->second;
    cur_page.dedup_flag = false;
    int ret = ioctl(userfault_fd, CMD_UPDATE_PAGEFAULT, *(unsigned long*)&va);
    if (ret == -1) {
        printf("ioctl CMD_UPDATE_PAGEFAULT failed\n");
        exit(-1);
    }
    
    pthread_mutex_lock(&g_shared_memory->mutex);

    bool found = false;
    SharedPageInfo* shared_page = nullptr;
    
    //TODO: 一开始先在本地找，再在全局找, 实现全局懒加载
    for (int i = 0; i < MAX_DEDUPE_PAGES; i++) {
        if (!g_shared_memory->pages[i].valid) continue;
        
        if (strncmp(g_shared_memory->pages[i].hash, hash, HASH_SIZE) == 0) {
            found = true;
            shared_page = &g_shared_memory->pages[i];
            break;
        }
    }

    // printf("[%s] try_deduplicate_page(va=0x%llx, src=%p, size=%s, owner_va=%llx, found=%d, hash=%s)\n", 
    //        current_time().c_str(), 
    //        (unsigned long long)va,
    //        data,
    //        format_bytes(size).c_str(),
    //        found?shared_page->owner_va:NULL,
    //        found,
    //        hash);

    if (found) {
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
        accessDesc.location.id = 0;
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
        strcpy(cur_page.hash,hash);

        {
            ScopedTimer t_hook(API_CU_MEM_RELEASE_ORIG); 
            orig_cuMemRelease(old_handle);
            g_allocations_create.erase(old_handle);
        }

        g_saved_memory += size;
        g_current_allocated -= size;
        for(int i=0;i<MAX_SHARED_REFS;i++){
            if(shared_page->ref_pids[i]==0) {
                shared_page->ref_pids[i] = g_dedup_pid;
                shared_page->ref_vas[i] = va;
                break;
            } 
        }
        shared_page->ref_count++;
        printf("[%s] Deduplicated page at VA: 0x%llx, Size: %s, Saved: %s, pid: %d\n",
               current_time().c_str(),
               (unsigned long long)va,
               format_bytes(size).c_str(),
               format_bytes(g_saved_memory).c_str(),
               getpid());

        pthread_mutex_unlock(&g_shared_memory->mutex);
        return true;
    } else {
        ShareableHandle shareable_handle;
        CUresult export_res;
        {
            ScopedTimer t_hook(API_CU_MEM_EXPORT_ORIG); 
            export_res = orig_cuMemExportToShareableHandle(
                (void*)&shareable_handle,
                cur_page.handle,
                CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                0
            );
        }

        if (export_res == CUDA_SUCCESS) {
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
            // printf("export handle va=%lx hash=%s handle=%lx\n", va, hash, cur_page.handle);
        } else {
            // 错误处理
            const char* errorName, *errorString;
            cuGetErrorName(export_res, &errorName);
            cuGetErrorString(export_res, &errorString);
            fprintf(stderr, "Export failed: %s (%s) va=%lx hash=%s handle=%lx\n", errorName, errorString, va, hash, cur_page.handle);
        }
    }
    pthread_mutex_unlock(&g_shared_memory->mutex);
    return false;
}

extern "C" {
// ========================================
// VMM API Hook 实现
// ========================================

CUresult CUDAAPI cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytesize) {
    if (userfault_fd == -1) {
        printf("Initializing userfault...\n");
        init_userfault();
        printf("userfault initialized\n");
    }

    CUresult res = CUDA_ERROR_UNKNOWN;
    if(bytesize < g_page_size) {
        // printf("[%s] cuMemAlloc_v2(dptr=0x%llx, size=%s)\n", 
        //     current_time().c_str(), 
        //     (unsigned long long)*dptr,
        //     format_bytes(bytesize).c_str());
        res = orig_cuMemAlloc_v2? 
                    orig_cuMemAlloc_v2(dptr, bytesize) : 
                    CUDA_ERROR_NOT_INITIALIZED;
        if (res != CUDA_SUCCESS) 
            printf("hook in cuMemAlloc_v2, res %d\n",res);
        return res;
    }
    
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
    res = orig_cuMemAddressReserve(&va, aligned_size, 0, 0, 0);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "cuMemAddressReserve failed: %d\n", res);
        return res;
    }

    for (size_t i = 0; i < num_pages; i++) {
        CUmemGenericAllocationHandle handle;
        CUdeviceptr va_ptr = va + i * g_page_size;

        res = orig_cuMemCreate(&handle, g_page_size, &prop, 0);
        if (res != CUDA_SUCCESS) {
            fprintf(stderr, "cuMemCreate failed: %d\n", res);
            return res;
        }

        res = orig_cuMemMap(va_ptr, g_page_size, 0, handle, 0);
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
    
    res = orig_cuMemSetAccess(va, aligned_size, &accessDesc, 1);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "cuMemSetAccess failed: %d\n", res);
        cuMemUnmap(va, aligned_size);
        return res;
    }

    alloc_info.va = va;
    g_allocations_vmm[va] = alloc_info;
    
    *dptr = va;
    printf("[%s] cuMemAlloc_v2(dptr=0x%llx, size=%s, aligned_size=%s)\n", 
        current_time().c_str(), 
        (unsigned long long)*dptr,
        format_bytes(bytesize).c_str(),
        format_bytes(aligned_size).c_str());

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
    printf("[%s] cuMemAlloc(dptr=0x%llx, aligned_size=%s)\n", 
        current_time().c_str(), 
        (unsigned long long)*dptr,
        format_bytes(aligned_size).c_str());
    
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount) {
    // 小于2MB的分配直接由原生API处理
    if(ByteCount < g_page_size) {
        CUresult res;
        res = orig_cuMemcpyHtoD_v2? 
                    orig_cuMemcpyHtoD_v2(dstDevice, srcHost, ByteCount) : 
                    CUDA_ERROR_NOT_INITIALIZED;
        if (res != CUDA_SUCCESS) 
            printf("hook in cuMemcpyHtoD_v2, res %d\n",res);
        return res;
    }

    g_cuMemCpy_size += ByteCount;

    // printf("[%s] cuMemcpyHtoD_v2(dst=0x%llx, src=%p, size=%s, total_size=%s)\n", 
    //        current_time().c_str(), 
    //        (unsigned long long)dstDevice,
    //        srcHost,
    //        format_bytes(ByteCount).c_str(),
    //        format_bytes(g_cuMemCpy_size).c_str());
    
    CUdeviceptr current_addr = dstDevice;
    int bytes_remaining = ByteCount;
    const char* current_src = static_cast<const char*>(srcHost);
    
    while (bytes_remaining > 0) {
        // printf("[%s] page info(current_addr=%p,current_src=%p,bytes_remaining=%d)\n",current_time().c_str(),current_addr,current_src,bytes_remaining);
        bool in_vmm = false;
        bool try_deduplicate = false;
        CUdeviceptr page_start = 0;
        size_t page_offset = 0;
        size_t page_bytes = 0;
        
        {
            //std::lock_guard<std::mutex> lock(g_mutex);
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
            // printf("page_offset=%d page_bytes=%d bytes_remaining=%d current_addr=%p\n",page_offset,page_bytes,bytes_remaining,current_addr);
        }
        
        if (!in_vmm) {
            CUresult res;
            res = orig_cuMemcpyHtoD_v2? 
                        orig_cuMemcpyHtoD_v2(current_addr, static_cast<const void*>(current_src), g_page_size) : 
                        CUDA_ERROR_NOT_INITIALIZED;
            if (res != CUDA_SUCCESS) {
                printf("hook in cuMemcpyHtoD_v2, res %d\n",res);
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

        if (page_offset == 0 && page_bytes == g_page_size) {
            try_deduplicate = try_deduplicate_page(page_start, g_page_size);
            // printf("try_deduplicate=%d  page_start=%lx  \n", try_deduplicate, page_start);
        }
        else {// TODO 部分覆盖的情况
        }

        CUresult res;
        if (!try_deduplicate) {
            // 执行实际的内存拷贝
            res = orig_cuMemcpyHtoD_v2? 
                        orig_cuMemcpyHtoD_v2(current_addr, static_cast<const void*>(current_src), page_bytes) : 
                        CUDA_ERROR_NOT_INITIALIZED;
            if (res != CUDA_SUCCESS) {
                printf("hook in cuMemcpyHtoD_v2, res %d\n",res);
                return res;
            }
            // 共享，复制完之后马上加上写保护
            if (page_offset == 0 && page_bytes == g_page_size) {
                CUmemAccessDesc accessDesc;
                memset(&accessDesc, 0, sizeof(accessDesc));
                accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                accessDesc.location.id = DEDUP_DEVICE;
                accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READ;
                CUresult res = orig_cuMemSetAccess(page_start, g_page_size, &accessDesc, 1);
                if (res != CUDA_SUCCESS) {
                    printf("cuMemSetAccess fail, res:%d\n", res);
                    return res;
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
    if(ByteCount < g_page_size) {
        CUresult res;
        res = orig_cuMemcpyHtoD? 
                    orig_cuMemcpyHtoD(dstDevice, srcHost, ByteCount) : 
                    CUDA_ERROR_NOT_INITIALIZED;
        if (res != CUDA_SUCCESS) 
            printf("hook in cuMemcpyHtoD, res %d\n",res);
        return res;
    }

    g_cuMemCpy_size += ByteCount;

    printf("[%s] cuMemcpyHtoD(dst=0x%llx, src=%p, size=%s, total_size=%s)\n", 
           current_time().c_str(), 
           (unsigned long long)dstDevice,
           srcHost,
           format_bytes(ByteCount).c_str(),
           format_bytes(g_cuMemCpy_size).c_str());
    
    CUdeviceptr current_addr = dstDevice;
    int bytes_remaining = ByteCount;
    const char* current_src = static_cast<const char*>(srcHost);
    
    while (bytes_remaining > 0) {
        // printf("[%s] page info(current_addr=%p,current_src=%p,bytes_remaining=%d)\n",current_time().c_str(),current_addr,current_src,bytes_remaining);
        bool in_vmm = false;
        bool try_deduplicate = false;
        CUdeviceptr page_start = 0;
        size_t page_offset = 0;
        size_t page_bytes = 0;
        
        {
            //std::lock_guard<std::mutex> lock(g_mutex);
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
            // printf("page_offset=%d page_bytes=%d bytes_remaining=%d current_addr=%p\n",page_offset,page_bytes,bytes_remaining,current_addr);
        }
        
        if (!in_vmm) {
            CUresult res;
            res = orig_cuMemcpyHtoD? 
                        orig_cuMemcpyHtoD(current_addr, static_cast<const void*>(current_src), g_page_size) : 
                        CUDA_ERROR_NOT_INITIALIZED;
            if (res != CUDA_SUCCESS) {
                printf("hook in cuMemcpyHtoD, res %d\n",res);
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

        if (page_offset == 0 && page_bytes == g_page_size) {
            try_deduplicate = try_deduplicate_page(page_start, g_page_size);
            // printf("try_deduplicate=%d  page_start=%lx  \n", try_deduplicate, page_start);
        }
        else {// TODO 部分覆盖的情况
        }

        CUresult res;
        if (!try_deduplicate) {
            // 执行实际的内存拷贝
            res = orig_cuMemcpyHtoD? 
                        orig_cuMemcpyHtoD(current_addr, static_cast<const void*>(current_src), page_bytes) : 
                        CUDA_ERROR_NOT_INITIALIZED;
            if (res != CUDA_SUCCESS) {
                printf("hook in cuMemcpyHtoD, res %d\n",res);
                return res;
            }
            // 共享，复制完之后马上加上写保护
            if (page_offset == 0 && page_bytes == g_page_size) {
                CUmemAccessDesc accessDesc;
                memset(&accessDesc, 0, sizeof(accessDesc));
                accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                accessDesc.location.id = DEDUP_DEVICE;
                accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READ;
                res = orig_cuMemSetAccess(page_start, g_page_size, &accessDesc, 1);
                if (res != CUDA_SUCCESS) {
                    printf("cuMemSetAccess fail, res:%d\n", res);
                    return res;
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
    //TODO copy之后再加写保护，不用async

    printf("[%s] cuMemcpyHtoDAsync(dst=0x%llx, size=%s, total_size=%s)\n", 
           current_time().c_str(), 
           (unsigned long long)dstDevice,
           format_bytes(ByteCount).c_str(),
           format_bytes(g_cuMemCpy_size).c_str());
    
    CUdeviceptr current_addr = dstDevice;
    int bytes_remaining = ByteCount;
    const char* current_src = static_cast<const char*>(srcHost);
    
    while (bytes_remaining > 0) {
        // printf("[%s] page info(current_addr=%p,current_src=%p,bytes_remaining=%d)\n",current_time().c_str(),current_addr,current_src,bytes_remaining);
        bool in_vmm = false;
        bool try_deduplicate = false;
        CUdeviceptr page_start = 0;
        size_t page_offset = 0;
        size_t page_bytes = 0;
        
        {
            // std::lock_guard<std::mutex> lock(g_mutex);
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
            // printf("page_offset=%d page_bytes=%d bytes_remaining=%d current_addr=%p\n",page_offset,page_bytes,bytes_remaining,current_addr);
        }
        
        if (!in_vmm) {
            CUresult res;
            res = orig_cuMemcpyHtoD_v2? 
                        orig_cuMemcpyHtoD_v2(current_addr, static_cast<const void*>(current_src), g_page_size) : 
                        CUDA_ERROR_NOT_INITIALIZED;
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
        
        // 执行实际的内存拷贝
        CUresult res;
        {
            ScopedTimer t_hook(API_CU_MEMCPY_HTOD_ASYNC_ORIG); 
            res = orig_cuMemcpyHtoD_v2? 
                        orig_cuMemcpyHtoD_v2(current_addr, static_cast<const void*>(current_src), page_bytes) : 
                        CUDA_ERROR_NOT_INITIALIZED;
        }
        if (res != CUDA_SUCCESS) {
            if (res == 1) {
                CUmemAccessDesc accessDesc;
                memset(&accessDesc, 0, sizeof(accessDesc));
                accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                accessDesc.location.id = DEDUP_DEVICE;
                accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                res = orig_cuMemSetAccess(current_addr, g_page_size, &accessDesc, 1);
                if (res != CUDA_SUCCESS) {
                    printf("cuMemSetAccess fail, res:%d\n", res);
                    return res;
                }
                res = orig_cuMemcpyHtoD_v2? 
                    orig_cuMemcpyHtoD_v2(current_addr, static_cast<const void*>(current_src), page_bytes) : 
                    CUDA_ERROR_NOT_INITIALIZED;
                if (res != CUDA_SUCCESS)
                    printf("hook in cuMemcpyHtoD_v2, res=%d current_addr=%lx\n", res, current_addr);
            } else return res;
        }
        if (page_offset == 0 && page_bytes == g_page_size) {
            CUmemAccessDesc accessDesc;
            memset(&accessDesc, 0, sizeof(accessDesc));
            accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            accessDesc.location.id = DEDUP_DEVICE;
            accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READ;
            res = orig_cuMemSetAccess(page_start, g_page_size, &accessDesc, 1);
            if (res != CUDA_SUCCESS) {
                printf("cuMemSetAccess fail, res:%d\n", res);
                return res;
            }
        }

        if (page_offset == 0 && page_bytes == g_page_size) {
            try_deduplicate = try_deduplicate_page(page_start, g_page_size);
            // printf("try_deduplicate=%d  page_start=%lx  \n", try_deduplicate, page_start);
        }

        // if (!try_deduplicate) {
        //     // 共享，复制完之后马上加上写保护
        // }
        
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
        // fprintf(stderr, "cuMemFree: Invalid pointer: 0x%llx\n", (unsigned long long)dptr);
        // return CUDA_ERROR_INVALID_VALUE;
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
            fprintf(stderr, "cuMemFreeAsync: page free error: 0x%llx\n", (unsigned long long)va_ptr);
            return CUDA_ERROR_INVALID_VALUE;
        }
    } 
    res = cuMemAddressFree(alloc_info.va, alloc_info.aligned_size);
    // 从全局映射中移除
    g_allocations_vmm.erase(it);
    return res;
}

void *local_handle = NULL;
CUresult CUDAAPI cuGetProcAddress_v2(
    const char* symbol, void** pfn, int  cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult* symbolStatus
) { 
    if (!local_handle)
	    local_handle = dlopen (hook_path, RTLD_NOW | RTLD_NODELETE);
    *pfn = NULL;
    *pfn = dlsym (local_handle, symbol);
    CUresult res = CUDA_ERROR_UNKNOWN;
    if (*pfn)
	    res = CUDA_SUCCESS;
    else
	    res = orig_cuGetProcAddress_v2 (symbol, pfn, cudaVersion, flags, symbolStatus);
    printf("[%s] cuGetProcAddress_v2 called for: %s\n", current_time().c_str(), symbol);
    
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
	        local_handle = dlopen (hook_path, RTLD_NOW | RTLD_NODELETE);
        *pfn = NULL;
        *pfn = dlsym (local_handle, symbol);
        printf("[%s] cuGetProcAddress called for: %s\n", current_time().c_str(), symbol);
        if (*pfn)
            res = CUDA_SUCCESS;  
    } else {
        // printf("[%s] cuGetProcAddress called for: %s\n", current_time().c_str(), symbol);
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
        // std::lock_guard<std::mutex> lock(g_mutex);
        size_t num_pages = aligned_size / g_page_size;
        g_total_allocated += aligned_size;
        g_current_allocated += aligned_size;
        g_allocations_create[*handle] = size;

        // // 输出调试信息
        // printf("[%s] cuMemCreate(handle=%p, size=%s, pages=%zu) - Total: %s, Current: %s\n", 
        //     current_time().c_str(), 
        //     *handle,
        //     format_bytes(aligned_size).c_str(),
        //     num_pages,
        //     format_bytes(g_total_allocated).c_str(),
        //     format_bytes(g_current_allocated).c_str());
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
        // printf("[%s] cuMemUnmap(va=0x%llx, size=%s)\n", 
        //        current_time().c_str(), 
        //        (unsigned long long)va,
        //        format_bytes(size).c_str());
        
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
    
    // printf("[%s] cuMemRelease(handle=0x%llx) - Current: %s\n", 
    //         current_time().c_str(), 
    //         (unsigned long long)handle,
    //         format_bytes(g_current_allocated).c_str());
    
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
    
    if (res == CUDA_SUCCESS) {
        // printf("[%s] cuMemAddressFree(ptr=0x%llx, size=%s)\n", 
        //        current_time().c_str(), 
        //        (unsigned long long)ptr,
        //        format_bytes(size).c_str());
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

