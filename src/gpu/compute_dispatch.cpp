#include "compute_dispatch.h"
#include <d3dcompiler.h>
#include <stdexcept>

namespace khanary::gpu {

ComputeDispatch::ComputeDispatch(D3D11Device& dev) : dev_(dev) {}

bool ComputeDispatch::set_shader(ID3DBlob* blob) {
    cs_.Reset();
    HRESULT hr = dev_.GetDevice()->CreateComputeShader(
        blob->GetBufferPointer(), blob->GetBufferSize(),
        nullptr, cs_.GetAddressOf());
    return SUCCEEDED(hr);
}

void ComputeDispatch::set_constants(const void* data, size_t size) {
    if (!cbuf_ || cbuf_size_ < size) {
        cbuf_.Reset();
        cbuf_size_ = (size + 15) & ~15; // 16-byte align
        D3D11_BUFFER_DESC bd{};
        bd.ByteWidth      = (UINT)cbuf_size_;
        bd.Usage          = D3D11_USAGE_DYNAMIC;
        bd.BindFlags      = D3D11_BIND_CONSTANT_BUFFER;
        bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        dev_.GetDevice()->CreateBuffer(&bd, nullptr, cbuf_.GetAddressOf());
    }
    if (!cbuf_) return;
    D3D11_MAPPED_SUBRESOURCE mapped{};
    if (SUCCEEDED(dev_.GetContext()->Map(cbuf_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
        memcpy(mapped.pData, data, size);
        dev_.GetContext()->Unmap(cbuf_.Get(), 0);
    }
    auto* cb = cbuf_.Get();
    dev_.GetContext()->CSSetConstantBuffers(0, 1, &cb);
}

void ComputeDispatch::run(
    const std::vector<ID3D11UnorderedAccessView*>& uavs,
    const std::vector<ID3D11ShaderResourceView*>&  srvs,
    uint32_t x, uint32_t y, uint32_t z)
{
    auto* ctx = dev_.GetContext();
    ctx->CSSetShader(cs_.Get(), nullptr, 0);
    if (!srvs.empty()) ctx->CSSetShaderResources(0, (UINT)srvs.size(), srvs.data());
    if (!uavs.empty()) {
        std::vector<UINT> init_counts(uavs.size(), 0);
        ctx->CSSetUnorderedAccessViews(0, (UINT)uavs.size(), uavs.data(), init_counts.data());
    }
    ctx->Dispatch(x, y, z);

    // Unbind
    std::vector<ID3D11UnorderedAccessView*> null_uavs(uavs.size(), nullptr);
    std::vector<ID3D11ShaderResourceView*>  null_srvs(srvs.size(), nullptr);
    std::vector<UINT> init_counts(uavs.size(), 0);
    if (!uavs.empty()) ctx->CSSetUnorderedAccessViews(0, (UINT)uavs.size(), null_uavs.data(), init_counts.data());
    if (!srvs.empty()) ctx->CSSetShaderResources(0, (UINT)srvs.size(), null_srvs.data());
    ctx->CSSetShader(nullptr, nullptr, 0);
}

void ComputeDispatch::run(
    const std::vector<TensorBuffer*>& outputs,
    const std::vector<TensorBuffer*>& inputs,
    uint32_t x, uint32_t y, uint32_t z)
{
    std::vector<ID3D11UnorderedAccessView*> uavs;
    std::vector<ID3D11ShaderResourceView*>  srvs;
    for (auto* tb : outputs) if (tb && tb->uav()) uavs.push_back(tb->uav());
    for (auto* tb : inputs)  if (tb && tb->srv()) srvs.push_back(tb->srv());
    run(uavs, srvs, x, y, z);
}

} // namespace khanary::gpu
