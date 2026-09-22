#pragma once
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <string>
#include <unordered_map>
#include <memory>

namespace khanary::klsl {

using Microsoft::WRL::ComPtr;

struct CompileResult {
    ComPtr<ID3DBlob> bytecode;
    ComPtr<ID3DBlob> errors;
    bool ok;
    std::string error_msg;
};

class D3DCompileBridge {
public:
    D3DCompileBridge();

    // Compile HLSL text → ID3DBlob
    // entry: shader entry point name (e.g. "main")
    // target: shader target (e.g. "cs_5_0")
    CompileResult compile(
        const std::string& source,
        const std::string& entry  = "main",
        const std::string& target = "cs_5_0");

    // Compile with caching: key = sha256(source)
    ID3DBlob* compile_cached(
        const std::string& source,
        const std::string& entry  = "main",
        const std::string& target = "cs_5_0");

    // Load a pre-compiled shader file (.cso)
    static CompileResult load_cso(const std::string& path);

    // Clear the blob cache
    void clear_cache();

private:
    std::unordered_map<std::string, ComPtr<ID3DBlob>> cache_;
};

} // namespace khanary::klsl
