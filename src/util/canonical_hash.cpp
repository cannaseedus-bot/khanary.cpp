#include "canonical_hash.h"
#include "sha256.h"
#include <cmath>

namespace khanary::util {
std::string canonical_hash(const JsonValue& value) {
    return "sha256:" + sha256_hex(stringify(value));
}
std::string cycle_identity_hash(const JsonValue& identity) {
    JsonObject fields;
    const auto* version = identity.get("topology_version");
    if (!version || !((version->is_string() && !version->as_string().empty()) ||
        (version->is_number() && std::isfinite(version->as_number()) && version->as_number() >= 0 &&
         version->as_number() <= 9007199254740991.0 && std::floor(version->as_number()) == version->as_number())))
        throw std::runtime_error("cycle identity: invalid topology_version");
    fields["topology_version"] = parse(stringify(*version));
    for (const char* key : {"topology_hash", "candidate_delta_hash", "certificate_hash"}) {
        const auto hash = identity.get_string(key);
        if (!hash || hash->size() != 71 || hash->rfind("sha256:", 0) != 0)
            throw std::runtime_error(std::string("cycle identity: invalid ") + key);
        for (size_t i = 7; i < hash->size(); ++i)
            if (!((*hash)[i] >= '0' && (*hash)[i] <= '9') && !((*hash)[i] >= 'a' && (*hash)[i] <= 'f'))
                throw std::runtime_error(std::string("cycle identity: invalid ") + key);
        fields[key] = json_string(*hash);
    }
    return canonical_hash(*json_object(std::move(fields)));
}
}
