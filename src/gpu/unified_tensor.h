#pragma once

#include "d3d11_device.h"
#include "tensor_buffer.h"
#include "dml_ops.h"
#include "../util/json.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace khanary::gpu {

// Logical tensor contract above TensorBuffer/DMLOps. TensorLifecycle is an
// execution/storage state only; it does not replace K'UHUL semantic phases.
enum class TensorDType : uint8_t { F32, F16, BF16, I32, I64, U8 };
enum class TensorLayout : uint8_t { RowMajor, ColMajor, NCHW, NHWC, NCW, NWC };
enum class TensorDevice : uint8_t { CPU, DirectML, D3D11, CSO, MMAP };
enum class TensorLifecycle : uint8_t { POP, SEK, CHEN, XUL, KNU };
enum class TensorBackend : uint8_t { Host, DirectML, D3D11_CSO, CPUFallback };

struct UnifiedTensorDesc {
    std::string id;
    std::vector<uint32_t> shape;
    TensorDType dtype = TensorDType::F32;
    TensorLayout layout = TensorLayout::RowMajor;
    TensorDevice device = TensorDevice::CPU;
    TensorLifecycle lifecycle = TensorLifecycle::POP;
    TensorBackend backend = TensorBackend::Host;
    std::string cso_id;
    std::string slot_id;
    std::string micronaut_id;
};

class UnifiedTensor {
public:
    UnifiedTensor() = default;
    explicit UnifiedTensor(UnifiedTensorDesc desc);

    const UnifiedTensorDesc& desc() const { return desc_; }
    const std::string& id() const { return desc_.id; }
    const std::vector<uint32_t>& shape() const { return desc_.shape; }
    TensorDType dtype() const { return desc_.dtype; }
    TensorDevice device() const { return desc_.device; }
    TensorLifecycle lifecycle() const { return desc_.lifecycle; }
    TensorBackend backend() const { return desc_.backend; }
    const std::string& cso_id() const { return desc_.cso_id; }
    const std::string& slot_id() const { return desc_.slot_id; }
    const std::string& micronaut_id() const { return desc_.micronaut_id; }

    size_t numel() const;
    size_t element_size() const;
    size_t byte_size() const;

    void resize_zero();
    bool upload_f32(const float* data, size_t count);
    bool upload_bytes(const void* data, size_t bytes);
    bool download_bytes(void* dst, size_t bytes) const;
    const std::vector<uint8_t>& host_bytes() const { return host_; }
    std::vector<uint8_t>& mutable_host_bytes() { return host_; }

    const float* data_f32() const;
    float* data_f32();

    void set_lifecycle(TensorLifecycle lifecycle) { desc_.lifecycle = lifecycle; }
    void set_device(TensorDevice device) { desc_.device = device; }
    void set_backend(TensorBackend backend) { desc_.backend = backend; }
    void set_cso_id(std::string cso_id) { desc_.cso_id = std::move(cso_id); }
    void set_slot_id(std::string slot_id) { desc_.slot_id = std::move(slot_id); }
    void set_micronaut_id(std::string micronaut_id) { desc_.micronaut_id = std::move(micronaut_id); }
    std::string compute_cso_id() const;

    bool ensure_gpu(D3D11Device& device);
    bool sync_from_gpu();
    void clear_gpu();
    std::shared_ptr<TensorBuffer> gpu_buffer() const { return gpu_; }

    util::JsonPtr to_json() const;
    static std::optional<UnifiedTensor> from_json(const util::JsonValue& value);
    bool validate() const;

private:
    UnifiedTensorDesc desc_;
    std::vector<uint8_t> host_;
    std::shared_ptr<TensorBuffer> gpu_;
};

class UnifiedTensorRegistry {
public:
    bool register_tensor(UnifiedTensor tensor, bool overwrite = false);
    bool remove(const std::string& id);
    bool has(const std::string& id) const;
    std::shared_ptr<UnifiedTensor> get(const std::string& id) const;
    std::vector<std::string> ids() const;
    util::JsonPtr info_json(const std::string& id) const;
    util::JsonPtr list_json() const;
    util::JsonPtr stats_json() const;

private:
    mutable std::mutex mtx_;
    std::unordered_map<std::string, std::shared_ptr<UnifiedTensor>> tensors_;
};

bool gemm_f32_unified(UnifiedTensor& A,
                      UnifiedTensor& B,
                      UnifiedTensor& C,
                      float alpha = 1.0f,
                      float beta = 0.0f,
                      D3D11Device* device = nullptr,
                      DMLOps* dml = nullptr,
                      std::string* backend_used = nullptr);

} // namespace khanary::gpu
