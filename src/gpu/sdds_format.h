#pragma once
#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

namespace khanary::gpu {
// CPU-only SDDS v1 validation. All words are little-endian. Q8 elem_count
// counts packed uint32 words (four int8 values), matching TensorBuffer.
struct SDDSChunk {
    std::string name;
    uint32_t elem_count = 0;
    uint32_t format = 0;
    std::vector<uint8_t> data;
};
uint32_t sdds_crc32(const uint8_t* data, size_t size);
// Either returns every validated chunk, or throws. No partial result or GPU IO.
std::vector<SDDSChunk> read_sdds(const std::string& path);
bool validate_sdds(const std::string& path);
}
