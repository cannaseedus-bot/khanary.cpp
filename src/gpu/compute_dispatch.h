#pragma once
#include "d3d11_device.h"
#include "tensor_buffer.h"
#include <vector>

namespace khanary::gpu {

class ComputeDispatch {
public:
    explicit ComputeDispatch(D3D11Device& dev);

    // Set compiled CS blob and create shader
    bool set_shader(ID3DBlob* blob);

    // Execute a compute dispatch
    // uavs: write targets, srvs: read-only inputs
    // Dispatches ceil(x/1), ceil(y/1), ceil(z/1) groups
    void run(
        const std::vector<ID3D11UnorderedAccessView*>& uavs,
        const std::vector<ID3D11ShaderResourceView*>&  srvs,
        uint32_t x, uint32_t y, uint32_t z);

    // Convenience: run with TensorBuffers
    void run(
        const std::vector<TensorBuffer*>& outputs,
        const std::vector<TensorBuffer*>& inputs,
        uint32_t x, uint32_t y, uint32_t z);

    // 1-D convenience for kernels written with fixed thread-group size (default 64).
    void run_for_elements(
        const std::vector<ID3D11UnorderedAccessView*>& uavs,
        const std::vector<ID3D11ShaderResourceView*>&  srvs,
        uint32_t element_count,
        uint32_t threads_per_group = 64u);

    // Update a constant buffer (slot 0)
    void set_constants(const void* data, size_t size);

private:
    D3D11Device&                      dev_;
    ComPtr<ID3D11ComputeShader>       cs_;
    ComPtr<ID3D11Buffer>              cbuf_;
    size_t                            cbuf_size_ = 0;
};

} // namespace khanary::gpu
