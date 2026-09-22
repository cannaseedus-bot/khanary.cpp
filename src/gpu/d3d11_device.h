#pragma once
#include <d3d11.h>
#include <dxgi.h>
#include <wrl/client.h>
#include <string>
#include <vector>

namespace khanary::gpu {

using Microsoft::WRL::ComPtr;

struct AdapterInfo {
    uint32_t    index;
    std::string description;
    size_t      dedicated_video_memory;
    size_t      shared_system_memory;
};

class D3D11Device {
public:
    D3D11Device();
    ~D3D11Device();

    // Initialize: enumerate adapters, prefer Intel HD 4600 by description
    // Falls back to default adapter if preferred not found
    bool init(const std::string& preferred_adapter_substr = "Intel");

    bool is_valid() const { return device_ != nullptr; }

    ID3D11Device*        GetDevice()  const { return device_.Get(); }
    ID3D11DeviceContext* GetContext() const { return context_.Get(); }
    IDXGIAdapter*        GetAdapter() const { return adapter_.Get(); }
    IDXGIFactory*        GetFactory() const { return factory_.Get(); }

    const AdapterInfo& adapter_info() const { return adapter_info_; }
    D3D_FEATURE_LEVEL   feature_level() const { return feature_level_; }

    // Enumerate all adapters
    static std::vector<AdapterInfo> enumerate_adapters();

private:
    ComPtr<IDXGIFactory>        factory_;
    ComPtr<IDXGIAdapter>        adapter_;
    ComPtr<ID3D11Device>        device_;
    ComPtr<ID3D11DeviceContext> context_;
    D3D_FEATURE_LEVEL           feature_level_ = D3D_FEATURE_LEVEL_11_0;
    AdapterInfo                 adapter_info_{};
};

} // namespace khanary::gpu
