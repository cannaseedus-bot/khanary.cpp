#include "../gpu/unified_tensor.h"
#include "conformance.h"

#include <cmath>
#include <string>
#include <vector>

using namespace khanary;

namespace {

gpu::UnifiedTensor make_f32_tensor(const std::string& id,
                                   std::vector<uint32_t> shape,
                                   const std::vector<float>& data) {
    gpu::UnifiedTensorDesc desc;
    desc.id = id;
    desc.shape = std::move(shape);
    desc.dtype = gpu::TensorDType::F32;
    desc.layout = gpu::TensorLayout::RowMajor;
    gpu::UnifiedTensor tensor(std::move(desc));
    if (!tensor.upload_f32(data.data(), data.size()))
        throw std::runtime_error("upload_f32 failed");
    return tensor;
}

} // namespace

int main() {
    return test::run("unified_tensor", [](test::Suite& suite) {
        {
            auto tensor = make_f32_tensor("A", {2, 3}, {1, 2, 3, 4, 5, 6});
            suite.check("tensor_numel", tensor.numel() == 6);
            suite.check("tensor_bytes", tensor.byte_size() == 24);
            suite.check("tensor_cso_stable", tensor.compute_cso_id() == tensor.compute_cso_id());
            const auto json = tensor.to_json();
            suite.check("tensor_json_shape", json && json->is_object() &&
                        json->get("shape") && json->get("shape")->is_array());
            suite.check("tensor_authority_boundary",
                        json->get_string("authority_boundary").value_or("") ==
                            "tensor_lifecycle_is_execution_state_not_semantic_authority");
        }

        {
            gpu::UnifiedTensorRegistry registry;
            auto tensor = make_f32_tensor("A", {2, 3}, {1, 2, 3, 4, 5, 6});
            suite.check("registry_register", registry.register_tensor(std::move(tensor)));
            suite.check("registry_duplicate_rejected",
                        !registry.register_tensor(make_f32_tensor("A", {1, 1}, {1.0f})));
            suite.check("registry_has", registry.has("A"));
            suite.check("registry_ids", registry.ids() == std::vector<std::string>{"A"});
            suite.check("registry_info", registry.info_json("A")->is_object());
            suite.check("registry_remove", registry.remove("A"));
            suite.check("registry_removed", !registry.has("A"));
            suite.check("registry_rejects_placeholder_tensor",
                        !registry.register_tensor(gpu::UnifiedTensor{}));
        }

        {
            auto A = make_f32_tensor("A", {2, 3}, {1, 2, 3, 4, 5, 6});
            auto B = make_f32_tensor("B", {3, 2}, {7, 8, 9, 10, 11, 12});
            gpu::UnifiedTensorDesc cdesc;
            cdesc.id = "C";
            gpu::UnifiedTensor C(std::move(cdesc));
            std::string backend;
            suite.check("gemm_cpu_fallback",
                        gpu::gemm_f32_unified(A, B, C, 1.0f, 0.0f,
                                              nullptr, nullptr, &backend));
            suite.check("gemm_backend", backend == "cpu-fallback");
            suite.check("gemm_lifecycle", C.lifecycle() == gpu::TensorLifecycle::KNU);
            const float* c = C.data_f32();
            const float expected[4] = {58.0f, 64.0f, 139.0f, 154.0f};
            bool values_ok = c != nullptr;
            for (size_t i = 0; values_ok && i < 4; ++i)
                values_ok = std::fabs(c[i] - expected[i]) < 1e-5f;
            suite.check("gemm_values", values_ok);
        }

        {
            util::JsonObject shape;
            (void)shape;
            util::JsonObject row;
            row["id"] = util::json_string("J");
            row["shape"] = util::json_array({util::json_number(2.0), util::json_number(2.0)});
            row["dtype"] = util::json_string("f32");
            row["data"] = util::json_array({
                util::json_number(1.0), util::json_number(2.0),
                util::json_number(3.0), util::json_number(4.0)
            });
            auto parsed = gpu::UnifiedTensor::from_json(*util::json_object(std::move(row)));
            suite.check("tensor_from_json", parsed.has_value());
            suite.check("tensor_from_json_numel", parsed && parsed->numel() == 4);
            suite.check("tensor_from_json_value",
                        parsed && std::fabs(parsed->data_f32()[3] - 4.0f) < 1e-6f);
        }

        {
            util::JsonObject row;
            row["id"] = util::json_string("tensor:tensor_slot:REASON-u");
            row["slot_id"] = util::json_string("tensor_slot:REASON-u");
            row["micronaut_id"] = util::json_string("REASON-µ");
            row["shape"] = util::json_array({util::json_number(1.0), util::json_number(3.0)});
            row["dtype"] = util::json_string("f32");
            row["lifecycle"] = util::json_string("SEK");
            row["data"] = util::json_array({
                util::json_number(0.25), util::json_number(0.5), util::json_number(1.0)
            });
            auto parsed = gpu::UnifiedTensor::from_json(*util::json_object(std::move(row)));
            suite.check("tensor_slot_from_json", parsed.has_value());
            suite.check("tensor_slot_id_roundtrip",
                        parsed && parsed->slot_id() == "tensor_slot:REASON-u");
            suite.check("tensor_slot_micronaut_roundtrip",
                        parsed && parsed->micronaut_id() == "REASON-µ");
            auto encoded = parsed ? parsed->to_json() : util::json_null();
            suite.check("tensor_slot_json_slot_id",
                        encoded->get_string("slot_id").value_or("") == "tensor_slot:REASON-u");
            suite.check("tensor_slot_json_authority_boundary",
                        encoded->get_string("authority_boundary").value_or("") ==
                            "tensor_lifecycle_is_execution_state_not_semantic_authority");
        }
    }, 25);
}
