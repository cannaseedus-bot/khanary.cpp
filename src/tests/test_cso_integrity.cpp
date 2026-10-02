#include "../gpu/cso_integrity.h"
#include "conformance.h"

#include <string>

using namespace khanary;

namespace {

std::string sample_kernel() {
    return "StructuredBuffer<float> A : register(t0);\n"
           "RWStructuredBuffer<float> C : register(u0);\n"
           "[numthreads(64,1,1)] void main(uint3 id : SV_DispatchThreadID) { C[id.x] = A[id.x]; }\n";
}

} // namespace

int main() {
    return test::run("cso_integrity", [](test::Suite& suite) {
        const std::string kernel = sample_kernel();
        const std::string wrapped = "<![CDATA[" + kernel + "]]>";
        const std::string cso = gpu::cso_id_for_source(kernel);

        const auto plain = gpu::validate_cso_payload(kernel, "");
        suite.check("plaintext_without_cso_allowed", plain.ok && !plain.cdata_present);

        const auto valid = gpu::validate_cso_payload(wrapped, cso);
        suite.check("cdata_valid_match", valid.ok && valid.cdata_present);
        suite.check("cdata_computed_cso", valid.computed_cso == cso);

        const auto mismatch = gpu::validate_cso_payload(wrapped, "cso://bad");
        suite.check("cdata_mismatch_rejected",
                    !mismatch.ok && mismatch.error == "CSO_INTEGRITY_MISMATCH");

        const auto missing = gpu::validate_cso_payload(wrapped, "");
        suite.check("cdata_missing_identity_rejected",
                    !missing.ok && missing.error == "CSO_ID_REQUIRED");

        const auto unterminated = gpu::validate_cso_payload("<![CDATA[" + kernel, cso);
        suite.check("cdata_unterminated_rejected",
                    !unterminated.ok && unterminated.error == "CDATA_UNTERMINATED");

        util::JsonObject manifest;
        manifest["kernel_source"] = util::json_string(wrapped);
        manifest["@id"] = util::json_string(cso);
        const auto manifest_ok = gpu::validate_cso_manifest(*util::json_object(std::move(manifest)));
        suite.check("manifest_valid_match", manifest_ok.ok);

        util::JsonObject bad_manifest;
        bad_manifest["shader_source"] = util::json_string(wrapped);
        bad_manifest["cso_id"] = util::json_string("cso://wrong");
        const auto manifest_bad = gpu::validate_cso_manifest(*util::json_object(std::move(bad_manifest)));
        suite.check("manifest_mismatch_rejected",
                    !manifest_bad.ok && manifest_bad.error == "CSO_INTEGRITY_MISMATCH");
    }, 8);
}
