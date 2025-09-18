#include "ipc.h"


int ipc_recv_fd(int sock) {
    struct msghdr msg = {0};
    struct iovec iov[1];
    char buf[1];
    
    union {
        struct cmsghdr cm;
        char control[CMSG_SPACE(sizeof(int))];
    } control_un;
    
    msg.msg_control = control_un.control;
    msg.msg_controllen = sizeof(control_un.control);
    
    iov[0].iov_base = buf;
    iov[0].iov_len = sizeof(buf);
    msg.msg_iov = iov;
    msg.msg_iovlen = 1;
    
    if (recvmsg(sock, &msg, 0) <= 0) {
        return -1;
    }
    
    struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
    if (!cmsg || cmsg->cmsg_level != SOL_SOCKET || 
        cmsg->cmsg_type != SCM_RIGHTS) {
        return -1;
    }
    
    int fd;
    memcpy(&fd, CMSG_DATA(cmsg), sizeof(fd));
    return fd;
}

// 发送文件描述符
bool ipc_send_fd(int sock, int fd) {
    struct msghdr msg = {0};
    struct iovec iov[1];
    char buf[1] = {'F'};
    
    union {
        struct cmsghdr cm;
        char control[CMSG_SPACE(sizeof(int))];
    } control_un;
    
    msg.msg_control = control_un.control;
    msg.msg_controllen = sizeof(control_un.control);
    
    struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_len = CMSG_LEN(sizeof(int));
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    memcpy(CMSG_DATA(cmsg), &fd, sizeof(fd));
    
    iov[0].iov_base = buf;
    iov[0].iov_len = sizeof(buf);
    msg.msg_iov = iov;
    msg.msg_iovlen = 1;
    
    return sendmsg(sock, &msg, 0) > 0;
}

// 获取当前进程的IPC套接字路径
std::string ipc_get_current_sock_path() {
    char path[108];
    snprintf(path, sizeof(path), "/tmp/cuda_dedup_%d.sock", getpid());
    return std::string(path);
}

void ipc_cleanup() {
    g_ipc_state.running = false;
    
    if (g_ipc_state.ipc_sock != -1) {
        shutdown(g_ipc_state.ipc_sock, SHUT_RDWR);
        close(g_ipc_state.ipc_sock);
        
        unlink(ipc_get_current_sock_path().c_str());
    }
    
    if (g_ipc_state.handler_thread) {
        pthread_join(g_ipc_state.handler_thread, NULL);
    }
}

// 从指定进程获取共享页面的文件描述符
int ipc_get_shared_fd(pid_t owner_pid, const char* sock_path, const char* hash) {
    int sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock < 0) {
        perror("socket creation failed");
        return -1;
    }

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, sock_path, sizeof(addr.sun_path)-1);

    if (connect(sock, (struct sockaddr*)&addr, sizeof(addr))) {
        perror("connect failed");
        close(sock);
        return -1;
    }

    char request = 'R';
    if (write(sock, &request, 1) != 1) {
        perror("write request failed");
        close(sock);
        return -1;
    }
    if (write(sock, hash, HASH_SIZE) != HASH_SIZE) {
        perror("write hash failed");
        close(sock);
        return -1;
    }

    // 接收文件描述符
    int fd = ipc_recv_fd(sock);
    close(sock);

    if (fd < 0) {
        printf("Failed to receive FD for hash %s from pid %d\n", hash, owner_pid);
        return -1;
    }
    
    return fd;
}

bool ipc_transfer_ownership(pid_t new_owner_pid, const IpcRequest* req, int shareable_fd) {
    char sock_path[108];
    snprintf(sock_path, sizeof(sock_path), "/tmp/cuda_dedup_%d.sock", new_owner_pid);
    
    int sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock < 0) {
        perror("socket creation failed");
        return false;
    }

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, sock_path, sizeof(addr.sun_path)-1);

    if (connect(sock, (struct sockaddr*)&addr, sizeof(addr))) {
        perror("connect failed");
        close(sock);
        return false;
    }

    char request_type = 'T';
    if (write(sock, &request_type, 1) != 1) {
        perror("write request type failed");
        close(sock);
        return false;
    }

    if (write(sock, req, sizeof(IpcRequest)) != sizeof(IpcRequest)) {
        perror("write request struct failed");
        close(sock);
        return false;
    }

    if (!ipc_send_fd(sock, shareable_fd)) {
        perror("send fd failed");
        close(sock);
        return false;
    }

    IpcResponse resp;
    if (read(sock, &resp, sizeof(IpcResponse)) != sizeof(IpcResponse)) {
        perror("read response failed");
        close(sock);
        return false;
    }

    close(sock);
    return resp.success;
}

bool ipc_send_dedup_request(pid_t target_pid, unsigned long long va, const char* hash) {
    printf("gsh in ipc_send_dedup_request, va=%lx, hash=%s\n", va, hash);
    // 获取目标进程的socket路径
    char sock_path[108];
    snprintf(sock_path, sizeof(sock_path), "/tmp/cuda_dedup_%d.sock", target_pid);
    
    // 创建socket并连接
    int sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock < 0) {
        perror("socket creation failed");
        return false;
    }

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, sock_path, sizeof(addr.sun_path)-1);

    if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("connect failed");
        close(sock);
        return false;
    }

    // 准备并发送重删请求
    DedupRequest request;
    request.va = va;
    strncpy(request.hash, hash, HASH_SIZE);
    request.src_pid = getpid();

    char request_type = 'D';
    if (write(sock, &request_type, 1) != 1) {
        perror("write request type failed");
        close(sock);
        return false;
    }

    if (write(sock, &request, sizeof(request)) != sizeof(request)) {
        perror("write request failed");
        close(sock);
        return false;
    }

    // 等待响应
    DedupResponse response;
    if (read(sock, &response, sizeof(response)) != sizeof(response)) {
        perror("read response failed");
        close(sock);
        return false;
    }

    close(sock);
    return response.success;
}