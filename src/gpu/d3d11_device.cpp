#include "d3d11_device.h"
#include <stdexcept>
#include <algorithm>
#include <cctype>
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

    auto lower = [](std::string s) {
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        return s;
    };

    const std::string hint = lower(preferred_adapter_substr);
    const bool prefer_dedicated =
        hint.empty() ||
        hint == "auto" ||
        hint == "dedicated" ||
        hint == "discrete";

    // Enumerate adapters and collect candidates.
    ComPtr<IDXGIAdapter> first_adapter;
    AdapterInfo first_info{};
    bool have_first = false;

    ComPtr<IDXGIAdapter> best_dedicated_adapter;
    AdapterInfo best_dedicated_info{};
    bool have_best_dedicated = false;

    ComPtr<IDXGIAdapter> hinted_adapter;
    AdapterInfo hinted_info{};
    bool have_hint_match = false;

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

        if (!have_first) {
            first_adapter = adapter;
            first_info = info;
            have_first = true;
        }

        if (!have_best_dedicated ||
            info.dedicated_video_memory > best_dedicated_info.dedicated_video_memory) {
            best_dedicated_adapter = adapter;
            best_dedicated_info = info;
            have_best_dedicated = true;
        }

        if (!hint.empty() && !prefer_dedicated) {
            const std::string lname = lower(name);
            if (!have_hint_match && lname.find(hint) != std::string::npos) {
                hinted_adapter = adapter;
                hinted_info = info;
                have_hint_match = true;
            }
        }
    }

    ComPtr<IDXGIAdapter> chosen;
    AdapterInfo chosen_info{};
    if (prefer_dedicated && have_best_dedicated) {
        chosen = best_dedicated_adapter;
        chosen_info = best_dedicated_info;
    } else if (have_hint_match) {
        chosen = hinted_adapter;
        chosen_info = hinted_info;
    } else if (have_best_dedicated) {
        chosen = best_dedicated_adapter;
        chosen_info = best_dedicated_info;
    } else if (have_first) {
        chosen = first_adapter;
        chosen_info = first_info;
    } else {
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
