#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace khanary::world {

enum class DataType { FLOAT32, FLOAT16, INT8, BINARY_SDR };
enum class DeviceResidency { HOST_RAM, GPU_VRAM_D3D11, GPU_VRAM_CSO };

// Passive GPU-buffer descriptor: describes the shape, dtype, and residency of
// a tensor allocation without owning it.  physical_buffer_allocation is an
// opaque backend handle (e.g. raw ID3D11Buffer*); it stays null until the
// substrate is explicitly promoted from HOST_RAM to a device.
struct TensorSubstrate {
    uint64_t              tensor_id                  = 0;
    std::vector<uint32_t> shape;                         // e.g. {N, stride_floats}
    std::vector<uint32_t> strides;
    DataType              dtype                      = DataType::FLOAT32;
    DeviceResidency       residency                  = DeviceResidency::HOST_RAM;
    std::string           provider_identity;             // matches GmlNode::provider_id
    void*                 physical_buffer_allocation = nullptr;
};

} // namespace khanary::world
