#include <sys/socket.h>
#include <cstring>
#include <cstdio>
#include <unistd.h>
#include <string>
#include <sys/un.h>
#include <sys/uio.h>
#include <unistd.h>
#include <mutex>
const int HASH_SIZE = 17;

struct IPCState {
    int ipc_sock;
    pthread_t handler_thread;
    bool running;
};

static IPCState g_ipc_state = {-1, 0, false};

enum IpcRequestType {
    OWNERSHIP_TRANSFER
};

struct IpcRequest {
    IpcRequestType type;
    pid_t src_pid;
    pid_t dst_pid;
    char hash[HASH_SIZE];
};

// IPC响应结构
struct IpcResponse {
    bool success;
};

void ipc_cleanup();
int ipc_recv_fd(int sock);
bool ipc_send_fd(int sock, int fd);
std::string ipc_get_current_sock_path();
int ipc_get_shared_fd(pid_t owner_pid, const char* sock_path, const char* hash);
bool ipc_transfer_ownership(pid_t new_owner_pid, const IpcRequest* req, int shareable_fd);