#pragma once

#include "../util/json.h"
#include <string>

namespace khanary::gpu {

struct CsoIntegrityResult {
    bool        ok = false;
    bool        cdata_present = false;
    std::string error;
    std::string computed_cso;
    std::string declared_cso;
};

// Returns the first complete CDATA payload when present. If a CDATA opening
// exists without a closing boundary, the returned string is empty and the
// caller should treat the payload as malformed.
std::string unwrap_cdata_source(const std::string& source);
bool has_cdata_open(const std::string& source);
bool has_cdata_close(const std::string& source);

std::string cso_id_for_source(const std::string& source);
CsoIntegrityResult validate_cso_payload(const std::string& kernel_source,
                                        const std::string& declared_cso);
CsoIntegrityResult validate_cso_manifest(const util::JsonValue& manifest);

} // namespace khanary::gpu
