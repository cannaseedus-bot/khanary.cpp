#pragma once
#include "json.h"
#include <filesystem>

namespace khanary::util {
// Local, offline schema subset used by data/schema. Unknown assertion keywords
// and external references fail closed; this is not a general JSON Schema engine.
// Returns false for invalid instances. Throws for missing/unsupported schemas.
bool validate_schema_file(const JsonValue& instance, const std::filesystem::path& schema,
                          std::string* error = nullptr);
}
