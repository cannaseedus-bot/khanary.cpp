#pragma once
#include "json.h"

namespace khanary::util {
// json-c14n-v1: sorted UTF-8 keys and compact, finite binary64 JSON. This is a
// repository contract, not an RFC 8785 claim. See docs/CONFORMANCE.md.
std::string canonical_hash(const JsonValue& value);
// Binds exactly the four fields in data/schema/cycle-hash.spec.json.
// Rejects malformed identity fields; unrelated run metadata is excluded.
std::string cycle_identity_hash(const JsonValue& identity);
}
