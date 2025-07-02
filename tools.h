#include <stdio.h>
#include <time.h>
#include <string>
#include <cstring>
#include <openssl/sha.h> // 用于计算SHA256哈希

std::string current_time();
std::string format_bytes(size_t bytes);
std::string sha256(const void* data, size_t size);