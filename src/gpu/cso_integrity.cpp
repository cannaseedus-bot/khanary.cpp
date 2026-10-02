#include "cso_integrity.h"
#include "../util/sha256.h"

namespace khanary::gpu {

namespace {

constexpr const char* kCdataOpen = "<![CDATA[";
constexpr const char* kCdataClose = "]]>";
constexpr const char* kCsoPrefix = "cso://";

} // namespace

bool has_cdata_open(const std::string& source) {
    return source.find(kCdataOpen) != std::string::npos;
}

bool has_cdata_close(const std::string& source) {
    return source.find(kCdataClose) != std::string::npos;
}

std::string unwrap_cdata_source(const std::string& source) {
    const size_t open = source.find(kCdataOpen);
    if (open == std::string::npos) return source;
    const size_t start = open + 9;
    const size_t close = source.find(kCdataClose, start);
    if (close == std::string::npos) return "";
    return source.substr(start, close - start);
}

std::string cso_id_for_source(const std::string& source) {
    return std::string(kCsoPrefix) + util::sha256_hex(source.data(), source.size());
}

CsoIntegrityResult validate_cso_payload(const std::string& kernel_source,
                                        const std::string& declared_cso)
{
    CsoIntegrityResult out;
    out.declared_cso = declared_cso;
    out.cdata_present = has_cdata_open(kernel_source);

    if (out.cdata_present && !has_cdata_close(kernel_source)) {
        out.error = "CDATA_UNTERMINATED";
        return out;
    }

    const std::string unwrapped = unwrap_cdata_source(kernel_source);
    if (unwrapped.empty()) {
        out.error = "KERNEL_SOURCE_EMPTY";
        return out;
    }

    out.computed_cso = cso_id_for_source(unwrapped);
    if (out.cdata_present && declared_cso.empty()) {
        out.error = "CSO_ID_REQUIRED";
        return out;
    }
    if (!declared_cso.empty() && declared_cso != out.computed_cso) {
        out.error = "CSO_INTEGRITY_MISMATCH";
        return out;
    }

    out.ok = true;
    return out;
}

CsoIntegrityResult validate_cso_manifest(const util::JsonValue& manifest) {
    CsoIntegrityResult out;
    if (!manifest.is_object()) {
        out.error = "MANIFEST_NOT_OBJECT";
        return out;
    }

    const std::string kernel_source =
        manifest.get_string("kernel_source").value_or(
            manifest.get_string("shader_source").value_or(""));
    const std::string declared =
        manifest.get_string("cso_id").value_or(
            manifest.get_string("@id").value_or(""));
    return validate_cso_payload(kernel_source, declared);
}

} // namespace khanary::gpu
