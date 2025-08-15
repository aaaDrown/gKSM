#include <dlfcn.h>
#include "cuda.h"
#include <stdio.h>
#include <string>
#include <map>
#include <mutex>
#include <vector>
#include <unordered_map>
#include <cstring>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/prctl.h>
#include "tools.h"
#include "ipc.h"

const int MAX_DEDUPE_PAGES = 20480;
const int HASH_SIZE = 17; // SHA256字符串长度
#define CMD_START_USERFAULT _IOW('L', 1, unsigned long)
#define CMD_UPDATE_PAGEFAULT _IOW('L', 4, unsigned long)
#define MAX_SHARED_REFS 32
#define DEDUP_DEVICE 0
typedef int ShareableHandle;
struct SharedPageInfo {
    char hash[HASH_SIZE];         
    pid_t owner_pid;   
    CUdeviceptr owner_va;
    char sock_path[108];         
    bool valid;                   
    int device_id;   
    int ref_count;      
    int ref_pids[MAX_SHARED_REFS];        
    CUdeviceptr ref_vas[MAX_SHARED_REFS];            
};

struct LocalPageInfo {
    CUdeviceptr va;                
    CUmemGenericAllocationHandle handle;          
    char hash[HASH_SIZE];  
    bool dedup_flag = false;        
};

struct SharedMemory {
    pthread_mutex_t mutex;
    int mutex_initialized; 
    int initialized; 
    int page_count;  
    SharedPageInfo pages[MAX_DEDUPE_PAGES]; 
};

struct fault_info {
    unsigned long fault_addr;
    pid_t pid;
    int fault_type;
};

typedef struct {
    std::vector<LocalPageInfo> pages;
    CUdeviceptr va;
    size_t aligned_size;
} VMMAllocInfo;

// (虚拟地址 -> 页) 有多个page副本,本地
std::map<CUdeviceptr, LocalPageInfo>* g_page_map = nullptr;
std::map<CUmemGenericAllocationHandle, size_t> g_allocations_create;
static SharedMemory* g_shared_memory = nullptr;
static std::map<CUdeviceptr, VMMAllocInfo> g_allocations_vmm;

int userfault_fd = -1;
std::mutex g_mutex; 
size_t g_total_allocated = 0;
size_t g_current_allocated = 0;
size_t g_saved_memory = 0; 
size_t g_page_size = 2 * 1024 * 1024; 
size_t g_cuMemCpy_size = 0;
pid_t g_dedup_pid;
const char * hook_path = "/home/ubuntu/dedup/build/libcuda.so.1";
const char * orig_path = "/usr/lib/x86_64-linux-gnu/libcuda-native.so";

void init_shared_memory();
void init_ipc();
void init_hook_api();
void* ipc_handler_thread(void* arg);
void init_userfault();
// nvcc matrix.cu -o build/matrix -ldl -lcuda -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -L/lib/x86_64-linux-gnu