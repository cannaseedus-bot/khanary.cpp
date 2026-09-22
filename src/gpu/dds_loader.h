#pragma once
#include "d3d11_device.h"
#include "tensor_buffer.h"
#include <string>
#include <vector>
#include <memory>
#include <cstdint>

namespace khanary::gpu {

// SDDS wire format:
//  [4] magic = "SDDS"
//  [4] version = 1
//  [4] num_chunks
//  per chunk:
//    [64] name (null-padded)
//    [4]  elem_count
//    [4]  format (0=F32, 1=Q8, 2=F16)
//    [4]  data_size (bytes)
//    [4]  crc32
//    [data_size] raw data bytes
struct DDSChunkHeader {
    char     name[64];
    uint32_t elem_count;
    uint32_t format;       // TensorFormat enum
    uint32_t data_size;
    uint32_t crc32;
};

struct DDSFileHeader {
    char     magic[4]; // "SDDS"
    uint32_t version;
    uint32_t num_chunks;
};

struct LoadedShard {
    std::string               name;
    std::unique_ptr<TensorBuffer> buffer;
    bool                      crc_ok;
};

class DDSLoader {
public:
    explicit DDSLoader(D3D11Device& dev);

    // Load a .scxqdds shard file → vector of GPU tensors
    std::vector<LoadedShard> load(const std::string& path);

    // Validate CRC without loading into GPU
    static bool validate_crc(const std::string& path);

private:
    D3D11Device& dev_;

    static uint32_t compute_crc32(const uint8_t* data, size_t len);
};

} // namespace khanary::gpu
