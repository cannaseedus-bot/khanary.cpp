#include "unified_tensor.h"
#include "../util/sha256.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <sstream>

namespace khanary::gpu {

namespace {

std::string dtype_name(TensorDType dtype) {
    switch (dtype) {
        case TensorDType::F32: return "f32";
        case TensorDType::F16: return "f16";
        case TensorDType::BF16: return "bf16";
        case TensorDType::I32: return "i32";
        case TensorDType::I64: return "i64";
        case TensorDType::U8: return "u8";
    }
    return "f32";
}

std::optional<TensorDType> parse_dtype(const std::string& value) {
    if (value == "f32") return TensorDType::F32;
    if (value == "f16") return TensorDType::F16;
    if (value == "bf16") return TensorDType::BF16;
    if (value == "i32") return TensorDType::I32;
    if (value == "i64") return TensorDType::I64;
    if (value == "u8") return TensorDType::U8;
    return std::nullopt;
}

std::string layout_name(TensorLayout layout) {
    switch (layout) {
        case TensorLayout::RowMajor: return "row_major";
        case TensorLayout::ColMajor: return "col_major";
        case TensorLayout::NCHW: return "nchw";
        case TensorLayout::NHWC: return "nhwc";
        case TensorLayout::NCW: return "ncw";
        case TensorLayout::NWC: return "nwc";
    }
    return "row_major";
}

std::optional<TensorLayout> parse_layout(const std::string& value) {
    if (value == "row_major") return TensorLayout::RowMajor;
    if (value == "col_major") return TensorLayout::ColMajor;
    if (value == "nchw") return TensorLayout::NCHW;
    if (value == "nhwc") return TensorLayout::NHWC;
    if (value == "ncw") return TensorLayout::NCW;
    if (value == "nwc") return TensorLayout::NWC;
    return std::nullopt;
}

std::string device_name(TensorDevice device) {
    switch (device) {
        case TensorDevice::CPU: return "cpu";
        case TensorDevice::DirectML: return "directml";
        case TensorDevice::D3D11: return "d3d11";
        case TensorDevice::CSO: return "cso";
        case TensorDevice::MMAP: return "mmap";
    }
    return "cpu";
}

std::optional<TensorDevice> parse_device(const std::string& value) {
    if (value == "cpu") return TensorDevice::CPU;
    if (value == "directml") return TensorDevice::DirectML;
    if (value == "d3d11") return TensorDevice::D3D11;
    if (value == "cso") return TensorDevice::CSO;
    if (value == "mmap") return TensorDevice::MMAP;
    return std::nullopt;
}

std::string lifecycle_name(TensorLifecycle lifecycle) {
    switch (lifecycle) {
        case TensorLifecycle::POP: return "POP";
        case TensorLifecycle::SEK: return "SEK";
        case TensorLifecycle::CHEN: return "CHEN";
        case TensorLifecycle::XUL: return "XUL";
        case TensorLifecycle::KNU: return "KNU";
    }
    return "POP";
}

std::optional<TensorLifecycle> parse_lifecycle(const std::string& value) {
    if (value == "POP") return TensorLifecycle::POP;
    if (value == "SEK") return TensorLifecycle::SEK;
    if (value == "CHEN" || value == "Ch'en") return TensorLifecycle::CHEN;
    if (value == "XUL") return TensorLifecycle::XUL;
    if (value == "KNU") return TensorLifecycle::KNU;
    return std::nullopt;
}

std::string backend_name(TensorBackend backend) {
    switch (backend) {
        case TensorBackend::Host: return "host";
        case TensorBackend::DirectML: return "khanary-directml";
        case TensorBackend::D3D11_CSO: return "d3d11-cso";
        case TensorBackend::CPUFallback: return "cpu-fallback";
    }
    return "host";
}

std::optional<TensorBackend> parse_backend(const std::string& value) {
    if (value == "host") return TensorBackend::Host;
    if (value == "khanary-directml" || value == "directml") return TensorBackend::DirectML;
    if (value == "d3d11-cso") return TensorBackend::D3D11_CSO;
    if (value == "cpu-fallback" || value == "cpu") return TensorBackend::CPUFallback;
    return std::nullopt;
}

util::JsonArray shape_json(const std::vector<uint32_t>& shape) {
    util::JsonArray out;
    for (uint32_t dim : shape) out.push_back(util::json_number(static_cast<double>(dim)));
    return out;
}

} // namespace

UnifiedTensor::UnifiedTensor(UnifiedTensorDesc desc) : desc_(std::move(desc)) {
    resize_zero();
}

size_t UnifiedTensor::numel() const {
    if (desc_.shape.empty()) return 0;
    size_t out = 1;
    for (uint32_t dim : shape()) {
        if (dim == 0) return 0;
        if (out > std::numeric_limits<size_t>::max() / dim) return 0;
        out *= dim;
    }
    return out;
}

size_t UnifiedTensor::element_size() const {
    switch (desc_.dtype) {
        case TensorDType::F32: return 4;
        case TensorDType::F16:
        case TensorDType::BF16: return 2;
        case TensorDType::I32: return 4;
        case TensorDType::I64: return 8;
        case TensorDType::U8: return 1;
    }
    return 4;
}

size_t UnifiedTensor::byte_size() const {
    return numel() * element_size();
}

void UnifiedTensor::resize_zero() {
    host_.assign(byte_size(), uint8_t{0});
    desc_.cso_id.clear();
    gpu_.reset();
}

bool UnifiedTensor::upload_f32(const float* data, size_t count) {
    if (desc_.dtype != TensorDType::F32 || !data || count != numel()) return false;
    host_.resize(byte_size());
    std::memcpy(host_.data(), data, byte_size());
    desc_.cso_id.clear();
    if (gpu_) gpu_->upload(host_.data(), host_.size());
    return true;
}

bool UnifiedTensor::upload_bytes(const void* data, size_t bytes) {
    if (!data || bytes != byte_size()) return false;
    host_.resize(bytes);
    std::memcpy(host_.data(), data, bytes);
    desc_.cso_id.clear();
    if (gpu_) gpu_->upload(host_.data(), host_.size());
    return true;
}

bool UnifiedTensor::download_bytes(void* dst, size_t bytes) const {
    if (!dst || bytes != byte_size() || host_.size() != bytes) return false;
    std::memcpy(dst, host_.data(), bytes);
    return true;
}

const float* UnifiedTensor::data_f32() const {
    if (desc_.dtype != TensorDType::F32 || host_.size() != byte_size()) return nullptr;
    return reinterpret_cast<const float*>(host_.data());
}

float* UnifiedTensor::data_f32() {
    if (desc_.dtype != TensorDType::F32 || host_.size() != byte_size()) return nullptr;
    desc_.cso_id.clear();
    return reinterpret_cast<float*>(host_.data());
}

std::string UnifiedTensor::compute_cso_id() const {
    std::string material;
    material.reserve(host_.size() + 128);
    material += dtype_name(desc_.dtype);
    material += "|" + layout_name(desc_.layout);
    for (uint32_t dim : desc_.shape) {
        material += "|" + std::to_string(dim);
    }
    material += "|";
    material.append(reinterpret_cast<const char*>(host_.data()), host_.size());
    return "tensor:sha256:" + util::sha256_hex(material);
}

bool UnifiedTensor::ensure_gpu(D3D11Device& device) {
    if (desc_.dtype != TensorDType::F32) return false;
    if (gpu_ && gpu_->is_valid()) return true;
    TensorDesc gpu_desc;
    gpu_desc.name = desc_.id.empty() ? "unified_tensor" : desc_.id;
    gpu_desc.format = TensorFormat::F32;
    gpu_desc.elem_count = static_cast<uint32_t>(numel());
    gpu_desc.stride_bytes = static_cast<uint32_t>(element_size());
    gpu_ = std::make_shared<TensorBuffer>(device, gpu_desc);
    if (!gpu_ || !gpu_->is_valid()) {
        gpu_.reset();
        return false;
    }
    return gpu_->upload(host_.data(), host_.size());
}

bool UnifiedTensor::sync_from_gpu() {
    if (!gpu_ || !gpu_->is_valid()) return false;
    host_.resize(byte_size());
    if (!gpu_->readback(host_.data(), host_.size())) return false;
    desc_.cso_id.clear();
    return true;
}

void UnifiedTensor::clear_gpu() {
    gpu_.reset();
}

util::JsonPtr UnifiedTensor::to_json() const {
    util::JsonObject out;
    out["@kind"] = util::json_string("khanary.unified_tensor.v1");
    out["id"] = util::json_string(desc_.id);
    out["shape"] = util::json_array(shape_json(desc_.shape));
    out["dtype"] = util::json_string(dtype_name(desc_.dtype));
    out["layout"] = util::json_string(layout_name(desc_.layout));
    out["device"] = util::json_string(device_name(desc_.device));
    out["lifecycle"] = util::json_string(lifecycle_name(desc_.lifecycle));
    out["backend"] = util::json_string(backend_name(desc_.backend));
    out["cso_id"] = util::json_string(desc_.cso_id.empty() ? compute_cso_id() : desc_.cso_id);
    if (!desc_.slot_id.empty()) out["slot_id"] = util::json_string(desc_.slot_id);
    if (!desc_.micronaut_id.empty()) out["micronaut_id"] = util::json_string(desc_.micronaut_id);
    out["numel"] = util::json_number(static_cast<double>(numel()));
    out["bytes"] = util::json_number(static_cast<double>(byte_size()));
    out["gpu_resident"] = util::json_bool(gpu_ && gpu_->is_valid());
    out["authority_boundary"] = util::json_string(
        "tensor_lifecycle_is_execution_state_not_semantic_authority");
    return util::json_object(std::move(out));
}

std::optional<UnifiedTensor> UnifiedTensor::from_json(const util::JsonValue& value) {
    if (!value.is_object()) return std::nullopt;
    const auto* shape_node = value.get("shape");
    if (!shape_node || !shape_node->is_array() || shape_node->as_array().empty())
        return std::nullopt;

    UnifiedTensorDesc desc;
    if (auto id = value.get_string("id")) desc.id = *id;
    for (const auto& dim : shape_node->as_array()) {
        if (!dim || !dim->is_number()) return std::nullopt;
        const double raw = dim->as_number();
        if (!std::isfinite(raw) || raw <= 0.0 || raw > 4294967295.0) return std::nullopt;
        desc.shape.push_back(static_cast<uint32_t>(raw));
    }
    if (auto dtype = value.get_string("dtype")) {
        auto parsed = parse_dtype(*dtype);
        if (!parsed) return std::nullopt;
        desc.dtype = *parsed;
    }
    if (auto layout = value.get_string("layout")) {
        auto parsed = parse_layout(*layout);
        if (!parsed) return std::nullopt;
        desc.layout = *parsed;
    }
    if (auto device = value.get_string("device")) {
        auto parsed = parse_device(*device);
        if (!parsed) return std::nullopt;
        desc.device = *parsed;
    }
    if (auto lifecycle = value.get_string("lifecycle")) {
        auto parsed = parse_lifecycle(*lifecycle);
        if (!parsed) return std::nullopt;
        desc.lifecycle = *parsed;
    }
    if (auto backend = value.get_string("backend")) {
        auto parsed = parse_backend(*backend);
        if (!parsed) return std::nullopt;
        desc.backend = *parsed;
    }
    if (auto cso = value.get_string("cso_id")) desc.cso_id = *cso;
    if (auto slot = value.get_string("slot_id")) desc.slot_id = *slot;
    if (auto micronaut = value.get_string("micronaut_id")) desc.micronaut_id = *micronaut;

    UnifiedTensor tensor(std::move(desc));
    if (const auto* data = value.get("data")) {
        if (!data->is_array()) return std::nullopt;
        if (tensor.dtype() != TensorDType::F32) return std::nullopt;
        if (data->as_array().size() != tensor.numel()) return std::nullopt;
        std::vector<float> host;
        host.reserve(data->as_array().size());
        for (const auto& item : data->as_array()) {
            if (!item || !item->is_number() || !std::isfinite(item->as_number()))
                return std::nullopt;
            host.push_back(static_cast<float>(item->as_number()));
        }
        if (!tensor.upload_f32(host.data(), host.size())) return std::nullopt;
    }
    if (!tensor.validate()) return std::nullopt;
    return tensor;
}

bool UnifiedTensor::validate() const {
    return !desc_.shape.empty() && numel() > 0 && byte_size() > 0;
}

bool UnifiedTensorRegistry::register_tensor(UnifiedTensor tensor, bool overwrite) {
    const std::string id = tensor.id();
    if (id.empty() || !tensor.validate()) return false;
    std::lock_guard<std::mutex> lk(mtx_);
    if (!overwrite && tensors_.count(id)) return false;
    tensors_[id] = std::make_shared<UnifiedTensor>(std::move(tensor));
    return true;
}

bool UnifiedTensorRegistry::remove(const std::string& id) {
    std::lock_guard<std::mutex> lk(mtx_);
    return tensors_.erase(id) > 0;
}

bool UnifiedTensorRegistry::has(const std::string& id) const {
    std::lock_guard<std::mutex> lk(mtx_);
    return tensors_.count(id) != 0;
}

std::shared_ptr<UnifiedTensor> UnifiedTensorRegistry::get(const std::string& id) const {
    std::lock_guard<std::mutex> lk(mtx_);
    auto it = tensors_.find(id);
    return it == tensors_.end() ? nullptr : it->second;
}

std::vector<std::string> UnifiedTensorRegistry::ids() const {
    std::lock_guard<std::mutex> lk(mtx_);
    std::vector<std::string> out;
    out.reserve(tensors_.size());
    for (const auto& [id, _] : tensors_) out.push_back(id);
    std::sort(out.begin(), out.end());
    return out;
}

util::JsonPtr UnifiedTensorRegistry::info_json(const std::string& id) const {
    auto tensor = get(id);
    return tensor ? tensor->to_json() : util::json_null();
}

util::JsonPtr UnifiedTensorRegistry::list_json() const {
    util::JsonObject out;
    out["@kind"] = util::json_string("khanary.unified_tensor_registry.v1");
    util::JsonArray ids_json;
    for (const auto& id : ids()) ids_json.push_back(util::json_string(id));
    out["ids"] = util::json_array(std::move(ids_json));
    out["authority_boundary"] = util::json_string(
        "registry_maps_tensors_not_semantic_authority");
    return util::json_object(std::move(out));
}

util::JsonPtr UnifiedTensorRegistry::stats_json() const {
    std::lock_guard<std::mutex> lk(mtx_);
    size_t bytes = 0;
    std::unordered_map<std::string, double> devices;
    std::unordered_map<std::string, double> backends;
    std::unordered_map<std::string, double> lifecycles;
    for (const auto& [_, tensor] : tensors_) {
        if (!tensor) continue;
        bytes += tensor->byte_size();
        ++devices[device_name(tensor->device())];
        ++backends[backend_name(tensor->backend())];
        ++lifecycles[lifecycle_name(tensor->lifecycle())];
    }

    const auto count_obj = [](const std::unordered_map<std::string, double>& values) {
        util::JsonObject out;
        for (const auto& [key, value] : values)
            out[key] = util::json_number(value);
        return util::json_object(std::move(out));
    };

    util::JsonObject out;
    out["@kind"] = util::json_string("khanary.unified_tensor_registry.stats.v1");
    out["tensor_count"] = util::json_number(static_cast<double>(tensors_.size()));
    out["bytes"] = util::json_number(static_cast<double>(bytes));
    out["devices"] = count_obj(devices);
    out["backends"] = count_obj(backends);
    out["lifecycles"] = count_obj(lifecycles);
    return util::json_object(std::move(out));
}

bool gemm_f32_unified(UnifiedTensor& A,
                      UnifiedTensor& B,
                      UnifiedTensor& C,
                      float alpha,
                      float beta,
                      D3D11Device* device,
                      DMLOps* dml,
                      std::string* backend_used)
{
    if (A.dtype() != TensorDType::F32 || B.dtype() != TensorDType::F32) return false;
    if (A.desc().layout != TensorLayout::RowMajor || B.desc().layout != TensorLayout::RowMajor)
        return false;
    if (A.shape().size() != 2 || B.shape().size() != 2) return false;

    const uint32_t M = A.shape()[0];
    const uint32_t K = A.shape()[1];
    const uint32_t K2 = B.shape()[0];
    const uint32_t N = B.shape()[1];
    if (K != K2 || M == 0 || N == 0 || K == 0) return false;

    if (C.dtype() != TensorDType::F32 || C.shape().size() != 2 ||
        C.shape()[0] != M || C.shape()[1] != N) {
        UnifiedTensorDesc desc;
        desc.id = C.id().empty() ? "C" : C.id();
        desc.shape = {M, N};
        desc.dtype = TensorDType::F32;
        desc.layout = TensorLayout::RowMajor;
        desc.device = TensorDevice::CPU;
        desc.backend = TensorBackend::Host;
        C = UnifiedTensor(std::move(desc));
    }

    if (device && device->is_valid() && dml &&
        A.ensure_gpu(*device) && B.ensure_gpu(*device) && C.ensure_gpu(*device)) {
        if (dml->gemm(A.gpu_buffer().get(), B.gpu_buffer().get(), C.gpu_buffer().get(),
                      M, K, N, alpha, beta) && C.sync_from_gpu()) {
            C.set_device(TensorDevice::DirectML);
            C.set_backend(TensorBackend::DirectML);
            C.set_lifecycle(TensorLifecycle::KNU);
            C.set_cso_id(C.compute_cso_id());
            if (backend_used) *backend_used = "khanary-directml";
            return true;
        }
    }

    const float* a = A.data_f32();
    const float* b = B.data_f32();
    float* c = C.data_f32();
    if (!a || !b || !c) return false;
    for (uint32_t i = 0; i < M; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < K; ++k)
                sum += a[static_cast<size_t>(i) * K + k] * b[static_cast<size_t>(k) * N + j];
            const size_t idx = static_cast<size_t>(i) * N + j;
            c[idx] = alpha * sum + beta * c[idx];
        }
    }
    C.clear_gpu();
    C.set_device(TensorDevice::CPU);
    C.set_backend(TensorBackend::CPUFallback);
    C.set_lifecycle(TensorLifecycle::KNU);
    C.set_cso_id(C.compute_cso_id());
    if (backend_used) *backend_used = "cpu-fallback";
    return true;
}

} // namespace khanary::gpu
