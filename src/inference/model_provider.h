#pragma once
#include <cstdint>
#include <string>

// ModelProvider — abstract execution contract for text generation.
//
// Two execution branches share one contract:
//   Local:  ModelRequest → LocalModelProvider → GGUF/SDDS/GPU → ModelResult
//   Remote: ModelRequest → CloudModelProvider → HTTP/SSE      → ModelResult
//
// Neither Sek nor Ch'en knows which branch fired.
// Policy (LOCAL-first / CLOUD-fallback / MISS) lives in ProviderRouter.

namespace khanary::inference {

// ── Request ──────────────────────────────────────────────────────────────────

struct ModelRequest {
    std::string  prompt;
    uint32_t     max_tokens   = 256;
    float        temperature  = 0.8f;
    float        top_p        = 0.95f;
    bool         streaming    = false;
    uint32_t     min_context  = 0;   // minimum context window required
};

// ── Result ───────────────────────────────────────────────────────────────────

enum class ModelStatus : uint8_t {
    OK      = 0,
    PARTIAL = 1,   // finish_reason == "length"
    ERROR   = 2,
    MISS    = 3,   // no provider was eligible/available
};

struct ModelUsage {
    uint32_t prompt_tokens    = 0;
    uint32_t generated_tokens = 0;
};

struct ModelResult {
    std::string  text;
    ModelStatus  status        = ModelStatus::OK;
    std::string  finish_reason;    // "stop" | "length" | "error"
    std::string  provider_id;      // "local" | "openai" | ...
    std::string  model_id;         // e.g. "gpt-oss-20b" | "gpt-4o"
    ModelUsage   usage;
    // Opaque JSON blob — passed through to Ch'en for episode recording.
    // Local providers populate fold/shard provenance; cloud providers
    // populate api_call_id / endpoint. Ch'en treats both as opaque.
    std::string  provenance_json;
};

inline ModelResult miss_result(const std::string& reason) {
    ModelResult r;
    r.status        = ModelStatus::MISS;
    r.finish_reason = reason;
    return r;
}

// ── Provider interface ───────────────────────────────────────────────────────

class ModelProvider {
public:
    virtual ~ModelProvider() = default;

    // Synchronous execute. Cloud providers block until the full response arrives
    // (or until max_tokens is reached). Streaming is a hint; providers may ignore it.
    virtual ModelResult  execute(const ModelRequest&) = 0;

    // True if the provider can accept requests right now (model loaded, creds valid, etc.).
    virtual bool         is_available()                     const = 0;

    // Stable identifier used by ProviderRouter and recorded in ModelResult.
    virtual std::string  provider_id()                      const = 0;

    // Capability probe. Common capability strings:
    //   "chat_completion"  "streaming"  "function_calling"
    //   "vision"           "code"       "context:N"   (N = minimum tokens)
    // Providers return false for unknown capabilities rather than throwing.
    virtual bool         supports(const std::string& cap)   const = 0;
};

// ── Router ───────────────────────────────────────────────────────────────────
//
// Selects the first available provider that satisfies required capabilities.
// Providers are checked in registration order (add() defines priority).
// execute() returns miss_result() if no provider is eligible.
//
// Required capabilities are derived from the request:
//   streaming    → "streaming"
//   min_context  → "context:N" (if N > 0)
//
// Cloud execution changes privacy/cost/network requirements — ProviderRouter
// enforces this at the policy boundary, not silently inside a provider.

class ProviderRouter {
public:
    void add(ModelProvider* p) { providers_.push_back(p); }

    ModelProvider* select(const ModelRequest& req) const {
        for (auto* p : providers_) {
            if (!p->is_available()) continue;
            if (req.streaming && !p->supports("streaming")) continue;
            if (req.min_context > 0) {
                auto cap = "context:" + std::to_string(req.min_context);
                if (!p->supports(cap)) continue;
            }
            return p;
        }
        return nullptr;
    }

    ModelResult execute(const ModelRequest& req) const {
        ModelProvider* p = select(req);
        if (!p) return miss_result("no eligible provider");
        return p->execute(req);
    }

private:
    std::vector<ModelProvider*> providers_;
};

}  // namespace khanary::inference
