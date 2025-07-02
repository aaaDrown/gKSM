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

    //if(!(unit_index >= 2 && size >= 2)) return std::string("error");
    
    char buffer[32];
    snprintf(buffer, sizeof(buffer), "%.2f %s", size, units[unit_index]);
    return std::string(buffer);
}

// 计算SHA256哈希
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