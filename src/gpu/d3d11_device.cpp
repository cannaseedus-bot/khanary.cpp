#include "d3d11_device.h"
#include <stdexcept>
#include <algorithm>
#include <cwchar>

namespace khanary::gpu {

static std::string wstr_to_str(const wchar_t* ws) {
    std::string s;
    while (*ws) s += (char)(unsigned char)*ws++;
    return s;
}

D3D11Device::D3D11Device() {}
D3D11Device::~D3D11Device() {}

std::vector<AdapterInfo> D3D11Device::enumerate_adapters() {
    std::vector<AdapterInfo> out;
    ComPtr<IDXGIFactory> factory;
    if (FAILED(CreateDXGIFactory(__uuidof(IDXGIFactory), (void**)factory.GetAddressOf())))
        return out;

    ComPtr<IDXGIAdapter> adapter;
    for (uint32_t i = 0; factory->EnumAdapters(i, adapter.ReleaseAndGetAddressOf()) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC desc{};
        adapter->GetDesc(&desc);
        AdapterInfo info;
        info.index = i;
        info.description = wstr_to_str(desc.Description);
        info.dedicated_video_memory = desc.DedicatedVideoMemory;
        info.shared_system_memory   = desc.SharedSystemMemory;
        out.push_back(std::move(info));
    }
    return out;
}

bool D3D11Device::init(const std::string& preferred_adapter_substr) {
    HRESULT hr = CreateDXGIFactory(__uuidof(IDXGIFactory), (void**)factory_.GetAddressOf());
    if (FAILED(hr)) return false;

    // Enumerate and pick preferred adapter
    ComPtr<IDXGIAdapter> chosen;
    AdapterInfo chosen_info{};
    uint32_t chosen_idx = 0;

    ComPtr<IDXGIAdapter> adapter;
    for (uint32_t i = 0; factory_->EnumAdapters(i, adapter.ReleaseAndGetAddressOf()) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC desc{};
        adapter->GetDesc(&desc);
        std::string name = wstr_to_str(desc.Description);

        AdapterInfo info;
        info.index = i;
        info.description = name;
        info.dedicated_video_memory = desc.DedicatedVideoMemory;
        info.shared_system_memory   = desc.SharedSystemMemory;

        // Prefer adapter matching the hint string (case-insensitive)
        bool matches = false;
        if (!preferred_adapter_substr.empty()) {
            std::string lname = name, lhint = preferred_adapter_substr;
            std::transform(lname.begin(), lname.end(), lname.begin(), [](unsigned char c){ return (char)std::tolower(c); });
            std::transform(lhint.begin(), lhint.end(), lhint.begin(), [](unsigned char c){ return (char)std::tolower(c); });
            matches = lname.find(lhint) != std::string::npos;
        }

        if (!chosen || matches) {
            chosen = adapter;
            chosen_info = info;
            chosen_idx = i;
            if (matches) break; // take the first match
        }
    }

    if (!chosen) {
        // Try default adapter
        if (FAILED(factory_->EnumAdapters(0, chosen.GetAddressOf())))
            return false;
    }

    adapter_ = chosen;
    adapter_info_ = chosen_info;

    D3D_FEATURE_LEVEL levels[] = { D3D_FEATURE_LEVEL_11_0 };
    UINT flags = 0;
#ifdef _DEBUG
    flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

    hr = D3D11CreateDevice(
        adapter_.Get(),
        D3D_DRIVER_TYPE_UNKNOWN,
        nullptr, flags,
        levels, 1,
        D3D11_SDK_VERSION,
        device_.GetAddressOf(),
        &feature_level_,
        context_.GetAddressOf()
    );

    return SUCCEEDED(hr);
}

} // namespace khanary::gpu
