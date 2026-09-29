#include "tensor_buffer.h"
#include <stdexcept>

namespace khanary::gpu {

static DXGI_FORMAT format_to_dxgi(TensorFormat f) {
    switch (f) {
        case TensorFormat::F32: return DXGI_FORMAT_R32_FLOAT;
        case TensorFormat::Q8:  return DXGI_FORMAT_R32_UINT;
        case TensorFormat::F16: return DXGI_FORMAT_R16_FLOAT;
    }
    return DXGI_FORMAT_R32_FLOAT;
}

static uint32_t format_stride(TensorFormat f) {
    switch (f) {
        case TensorFormat::F32: return 4;
        case TensorFormat::Q8:  return 4; // 4 INT8 packed per uint32
        case TensorFormat::F16: return 2;
    }
    return 4;
}

TensorBuffer::TensorBuffer(D3D11Device& dev, const TensorDesc& desc)
    : dev_(&dev), desc_(desc)
{
    if (desc_.stride_bytes == 0)
        desc_.stride_bytes = format_stride(desc_.format);
    byte_size_ = (size_t)desc_.elem_count * desc_.stride_bytes;
    if (byte_size_ == 0) return;
    create_buffers();
}

TensorBuffer::~TensorBuffer() {}

bool TensorBuffer::create_buffers() {
    auto* d = dev_->GetDevice();

    D3D11_BUFFER_DESC bd{};
    bd.ByteWidth           = (UINT)byte_size_;
    bd.Usage               = D3D11_USAGE_DEFAULT;
    bd.BindFlags           = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    bd.CPUAccessFlags      = 0;
    bd.MiscFlags           = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    bd.StructureByteStride = desc_.stride_bytes;

    if (FAILED(d->CreateBuffer(&bd, nullptr, buf_.GetAddressOf())))
        return false;

    // Staging buffer for readback — must match structured layout for CopyResource
    D3D11_BUFFER_DESC sd = bd;
    sd.Usage          = D3D11_USAGE_STAGING;
    sd.BindFlags      = 0;
    sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    sd.MiscFlags      = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    if (FAILED(d->CreateBuffer(&sd, nullptr, staging_.GetAddressOf())))
        return false;

    // SRV — structured buffers require BUFFER dimension with DXGI_FORMAT_UNKNOWN
    D3D11_SHADER_RESOURCE_VIEW_DESC srvd{};
    srvd.Format              = DXGI_FORMAT_UNKNOWN;
    srvd.ViewDimension       = D3D11_SRV_DIMENSION_BUFFER;
    srvd.Buffer.FirstElement = 0;
    srvd.Buffer.NumElements  = desc_.elem_count;
    if (FAILED(d->CreateShaderResourceView(buf_.Get(), &srvd, srv_.GetAddressOf())))
        return false;

    // UAV
    D3D11_UNORDERED_ACCESS_VIEW_DESC uavd{};
    uavd.Format              = DXGI_FORMAT_UNKNOWN;
    uavd.ViewDimension       = D3D11_UAV_DIMENSION_BUFFER;
    uavd.Buffer.FirstElement = 0;
    uavd.Buffer.NumElements  = desc_.elem_count;
    if (FAILED(d->CreateUnorderedAccessView(buf_.Get(), &uavd, uav_.GetAddressOf())))
        return false;

    return true;
}

bool TensorBuffer::upload(const void* data, size_t bytes) {
    if (!dev_ || !buf_) return false;
    if (bytes > byte_size_) bytes = byte_size_;
    auto* ctx = dev_->GetContext();

    D3D11_BOX box{ 0, 0, 0, (UINT)bytes, 1, 1 };
    ctx->UpdateSubresource(buf_.Get(), 0, &box, data, 0, 0);
    return true;
}

bool TensorBuffer::readback(void* dst, size_t bytes) {
    if (!dev_ || !buf_ || !staging_) return false;
    auto* ctx = dev_->GetContext();
    ctx->CopyResource(staging_.Get(), buf_.Get());

    D3D11_MAPPED_SUBRESOURCE mapped{};
    if (FAILED(ctx->Map(staging_.Get(), 0, D3D11_MAP_READ, 0, &mapped)))
        return false;
    if (bytes > byte_size_) bytes = byte_size_;
    memcpy(dst, mapped.pData, bytes);
    ctx->Unmap(staging_.Get(), 0);
    return true;
}

} // namespace khanary::gpu
