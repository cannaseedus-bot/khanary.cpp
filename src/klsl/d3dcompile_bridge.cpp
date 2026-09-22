#include "d3dcompile_bridge.h"
#include "../util/sha256.h"
#include <fstream>
#include <sstream>

#pragma comment(lib, "d3dcompiler.lib")

namespace khanary::klsl {

D3DCompileBridge::D3DCompileBridge() {}

CompileResult D3DCompileBridge::compile(
    const std::string& source,
    const std::string& entry,
    const std::string& target)
{
    CompileResult r{};
    UINT flags = D3DCOMPILE_OPTIMIZATION_LEVEL1;
#ifdef _DEBUG
    flags |= D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#endif
    HRESULT hr = D3DCompile(
        source.data(), source.size(),
        nullptr, nullptr, nullptr,
        entry.c_str(), target.c_str(),
        flags, 0,
        r.bytecode.GetAddressOf(),
        r.errors.GetAddressOf()
    );
    r.ok = SUCCEEDED(hr);
    if (!r.ok && r.errors) {
        r.error_msg = std::string(
            static_cast<const char*>(r.errors->GetBufferPointer()),
            r.errors->GetBufferSize());
    }
    return r;
}

ID3DBlob* D3DCompileBridge::compile_cached(
    const std::string& source,
    const std::string& entry,
    const std::string& target)
{
    std::string key = khanary::util::sha256_hex(source) + ":" + entry + ":" + target;
    auto it = cache_.find(key);
    if (it != cache_.end()) return it->second.Get();

    auto r = compile(source, entry, target);
    if (!r.ok) return nullptr;
    cache_[key] = r.bytecode;
    return cache_[key].Get();
}

CompileResult D3DCompileBridge::load_cso(const std::string& path) {
    CompileResult r{};
    std::ifstream f(path, std::ios::binary);
    if (!f) { r.ok = false; r.error_msg = "Cannot open " + path; return r; }
    std::ostringstream ss; ss << f.rdbuf();
    std::string data = ss.str();
    HRESULT hr = D3DCreateBlob(data.size(), r.bytecode.GetAddressOf());
    if (FAILED(hr)) { r.ok = false; return r; }
    memcpy(r.bytecode->GetBufferPointer(), data.data(), data.size());
    r.ok = true;
    return r;
}

void D3DCompileBridge::clear_cache() { cache_.clear(); }

} // namespace khanary::klsl
