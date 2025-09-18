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
#define CMD_START_USERFAULT _IOW('L', 1, unsigned long)
#define CMD_UPDATE_PAGEFAULT _IOW('L', 4, unsigned long)
#define MAX_SHARED_REFS 32
#define MAX_MEMCPY_LIMIT 3
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
    bool pagefault_flag = false; 
    
    // background scan   
    int memcpy_cnt = 0;
    char prev_hash[HASH_SIZE];
    bool is_stable = false;
    bool is_deduped = false;
    int unchanged_scans = 0; 

    bool is_exported = false;
};

struct CandidatePageInfo {
    char hash[HASH_SIZE];
    pid_t pid;
    CUdeviceptr va;
    bool valid;
    int unchanged_scans; 
};

struct SharedMemory {
    pthread_mutex_t mutex;
    int initialized;
    int shared_page_count;
    int candidate_page_count;
    SharedPageInfo shared_pages[MAX_DEDUPE_PAGES];
    CandidatePageInfo candidate_pages[MAX_DEDUPE_PAGES]; 
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

std::map<CUdeviceptr, LocalPageInfo>* g_page_map = nullptr;
static SharedMemory* g_shared_memory = nullptr;
static std::map<CUdeviceptr, VMMAllocInfo> g_allocations_vmm;

int userfault_fd = -1;
pid_t g_dedup_pid = -1;
std::mutex g_mutex; 
size_t g_current_allocated = 0;
size_t g_saved_memory = 0; 
size_t g_saved_memory_print_threshold = 500 * 1024 * 1024;
size_t g_page_size = 2 * 1024 * 1024; 
size_t g_cuMemCpy_size = 0;
size_t pages_deduped = 0;

size_t g_background_scans = 0;

bool g_background_scanning = false;
bool g_userfault_handling = false;
bool g_ipc_handling = false;
pthread_t background_scan;
pthread_t userfault_handler;
pthread_t ipc_handler;

const char * hook_path = "/home/ubuntu/dedup/build/libcuda.so.1";
const char * orig_path = "/usr/lib/x86_64-linux-gnu/libcuda-native.so";

HookConfig g_hook_config = {
    .enable_deduplication = true,
    .enable_background_scan = true,
    .initial_scan_interval_ms = 2000,
    .max_scan_interval_ms = 30000,
    .boot_scan_interval_ms = 30000,
    .stable_threshold = 3,
    .max_batch_size = 16,
    .verbose_logging = false,
    .enable_statistics = true,
    .userfault_fd_path = "/dev/dedup_dev",
    .device_id = 0,
    .page_size = 2097152,
    .shared_memory_name = "/cuda_dedup_shared",
    .max_dedupe_pages = 20480,
    .max_shared_refs = 32
};

void init_shared_memory();
void init_ipc();
void init_hook_api();
void* ipc_handler_thread(void* arg);
void init_userfault();
void init_background_scan();
bool try_deduplicate_page(CUdeviceptr va, const void * data, size_t size, const char* hash);
