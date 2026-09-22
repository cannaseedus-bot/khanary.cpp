#pragma once
#include "d3d11_device.h"
#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

namespace khanary::gpu {

enum class TensorFormat {
    F32,    // R32_FLOAT, 4 bytes/element
    Q8,     // R32_UINT  packed, 4 INT8 per uint32
    F16,    // R16_FLOAT
};

struct TensorDesc {
    std::string name;
    TensorFormat format = TensorFormat::F32;
    uint32_t     elem_count = 0;
    uint32_t     stride_bytes = 4;
};

class TensorBuffer {
public:
    TensorBuffer() = default;
    TensorBuffer(D3D11Device& dev, const TensorDesc& desc);
    ~TensorBuffer();

    bool is_valid() const { return buf_ != nullptr; }

    // Upload host data to GPU buffer
    bool upload(const void* data, size_t bytes);

    // Readback GPU buffer to host
    bool readback(void* dst, size_t bytes);

    ID3D11Buffer*             buffer()  const { return buf_.Get(); }
    ID3D11ShaderResourceView* srv()     const { return srv_.Get(); }
    ID3D11UnorderedAccessView* uav()    const { return uav_.Get(); }

    const TensorDesc& desc() const { return desc_; }
    size_t byte_size() const { return byte_size_; }

private:
    D3D11Device*                      dev_ = nullptr;
    TensorDesc                        desc_;
    size_t                            byte_size_ = 0;
    ComPtr<ID3D11Buffer>              buf_;
    ComPtr<ID3D11Buffer>              staging_;
    ComPtr<ID3D11ShaderResourceView>  srv_;
    ComPtr<ID3D11UnorderedAccessView> uav_;

    bool create_buffers();
};

} // namespace khanary::gpu
