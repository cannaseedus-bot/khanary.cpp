#include "dds_loader.h"
#include "sdds_format.h"
#include <stdexcept>

namespace khanary::gpu {
uint32_t DDSLoader::compute_crc32(const uint8_t* data, size_t len) {
    return sdds_crc32(data, len);
}
DDSLoader::DDSLoader(D3D11Device& dev) : dev_(dev) {}
std::vector<LoadedShard> DDSLoader::load(const std::string& path) {
    // Validate the complete file before allocating or exposing any GPU tensor.
    auto chunks = read_sdds(path);
    std::vector<LoadedShard> out;
    for (const auto& chunk : chunks) {
        TensorDesc desc;
        desc.name = chunk.name;
        desc.format = chunk.format == 1 ? TensorFormat::Q8 :
                      chunk.format == 2 ? TensorFormat::F16 : TensorFormat::F32;
        desc.elem_count = chunk.elem_count;
        desc.stride_bytes = chunk.format == 2 ? 2 : 4;
        auto tensor = std::make_unique<TensorBuffer>(dev_, desc);
        if (!tensor->is_valid() || !tensor->srv() || !tensor->uav() ||
            !tensor->upload(chunk.data.data(), chunk.data.size()))
            throw std::runtime_error("SDDS: GPU tensor allocation/upload failed");
        LoadedShard shard;
        shard.name = chunk.name;
        shard.buffer = std::move(tensor);
        shard.crc_ok = true;
        out.push_back(std::move(shard));
    }
    return out;
}
bool DDSLoader::validate_crc(const std::string& path) {
    return validate_sdds(path);
}
}
