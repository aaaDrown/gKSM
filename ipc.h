#include <sys/socket.h>
#include <cstring>
#include <cstdio>
#include <unistd.h>
#include <string>
#include <sys/un.h>
#include <sys/uio.h>
#include <unistd.h>
#include <mutex>


// IPC模块状态
struct IPCState {
    int ipc_sock;
    pthread_t handler_thread;
    bool running;
};

static IPCState g_ipc_state = {-1, 0, false};

void ipc_cleanup();
int ipc_recv_fd(int sock);
bool ipc_send_fd(int sock, int fd);
std::string ipc_get_current_sock_path();
int ipc_get_shared_fd(pid_t owner_pid, const char* sock_path, const char* hash);
