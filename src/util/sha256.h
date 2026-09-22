#pragma once
#include <string>
#include <cstdint>
#include <cstddef>
#include <array>

namespace khanary::util {

std::array<uint8_t, 32> sha256_bytes(const void* data, size_t len);
std::string sha256_hex(const void* data, size_t len);
std::string sha256_hex(const std::string& s);

} // namespace khanary::util
