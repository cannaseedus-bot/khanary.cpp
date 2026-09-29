#include "projection.h"
#include "../util/json.h"
#include "../util/sha256.h"
#include <algorithm>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace khanary::io {

ProjectionEngine::ProjectionEngine(const std::string& abi_book_path)
    : book_(load_book(abi_book_path))
{}

bool ProjectionEngine::has_abi(const std::string& version) const {
    return book_.version == version;
}

ProjectionEngine::AbiBook ProjectionEngine::load_book(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("ProjectionEngine: cannot open ABI book: " + path);

    std::string text(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>{});
    auto root = util::parse(text);
    if (!root || !root->is_object())
        throw std::runtime_error("ProjectionEngine: invalid JSON in: " + path);

    // Locate the first key starting with "⟁PROJECTION_ABI"
    const util::JsonValue* book_val  = nullptr;
    std::string            found_ver;
    for (const auto& [k, v] : root->as_object()) {
        if (k.rfind("\xe2\x9f\x81PROJECTION_ABI", 0) == 0 ||
            k.rfind("⟁PROJECTION_ABI", 0) == 0) {
            book_val  = v.get();
            found_ver = k;
            break;
        }
    }
    if (!book_val || !book_val->is_object())
        throw std::runtime_error(
            "ProjectionEngine: no ⟁PROJECTION_ABI_vN key found in: " + path);

    const auto& book_obj = book_val->as_object();
    AbiBook ab;
    ab.version = found_ver;

    // canonical_dims — the stable output ordering
    {
        auto it = book_obj.find("dimensions");
        if (it != book_obj.end() && it->second->is_array())
            for (const auto& dv : it->second->as_array())
                if (dv->is_string()) ab.canonical_dims.push_back(dv->as_string());
    }

    // event_to_dim — reverse index from authorized_events
    {
        auto it = book_obj.find("authorized_events");
        if (it != book_obj.end() && it->second->is_object()) {
            for (const auto& [dim, arr_val] : it->second->as_object()) {
                if (dim == "NOTE" || !arr_val->is_array()) continue;
                for (const auto& ev : arr_val->as_array())
                    if (ev->is_string()) ab.event_to_dim[ev->as_string()] = dim;
            }
        }
    }

    // Helper: load a string → set<string> policy map
    auto load_policy = [&](const std::string& key)
        -> std::unordered_map<std::string, std::unordered_set<std::string>>
    {
        std::unordered_map<std::string, std::unordered_set<std::string>> out;
        auto it = book_obj.find(key);
        if (it == book_obj.end() || !it->second->is_object()) return out;
        for (const auto& [policy_key, arr_val] : it->second->as_object()) {
            if (!arr_val->is_array()) continue;
            for (const auto& dv : arr_val->as_array())
                if (dv->is_string()) out[policy_key].insert(dv->as_string());
        }
        return out;
    };

    ab.thinking_dims = load_policy("thinking_retention");
    ab.target_dims   = load_policy("target_exposure");
    return ab;
}

ProjectionResult ProjectionEngine::project_v(
    const JROMTrace&         trace,
    const ProjectionRequest& request) const
{
    ProjectionResult res;
    res.abi_version    = request.abi_version;
    res.target         = request.target;
    res.thinking_level = request.thinking;
    res.anchor         = request.anchor;
    res.semantic_attention_contract = "Q=query_task_vector;K=semantic_key_vector_meta;V=book_payload";
    res.semantic_attention_formula = "S=W*C*cos(Q,K)";
    res.ok             = false;
    res.anchor_satisfied = false;

    if (book_.version != request.abi_version) {
        res.error = "ABI version mismatch: loaded=" + book_.version
                  + " requested=" + request.abi_version;
        return res;
    }

    // Compute allowed dims: thinking_retention ∩ target_exposure
    auto t_it = book_.thinking_dims.find(request.thinking);
    auto x_it = book_.target_dims.find(request.target);
    if (t_it == book_.thinking_dims.end()) {
        res.error = "Unknown thinking level: " + request.thinking;
        return res;
    }
    if (x_it == book_.target_dims.end()) {
        res.error = "Unknown target: " + request.target;
        return res;
    }
    std::unordered_set<std::string> allowed;
    for (const auto& d : t_it->second)
        if (x_it->second.count(d)) allowed.insert(d);

    // Anchor must be in the allowed set if specified
    if (!request.anchor.empty() && !allowed.count(request.anchor)) {
        res.error = "Anchor '" + request.anchor
                  + "' not allowed for thinking=" + request.thinking
                  + " target=" + request.target;
        return res;
    }

    // Bucket authorized events into their dimensions
    std::unordered_map<std::string, std::vector<const JROMEvent*>> dim_events;
    for (const auto& dim : allowed) dim_events[dim] = {};  // all allowed dims present

    for (const auto& ev : trace) {
        auto m_it = book_.event_to_dim.find(ev.type);
        if (m_it == book_.event_to_dim.end()) continue;
        const std::string& dim = m_it->second;
        if (!allowed.count(dim)) continue;
        dim_events[dim].push_back(&ev);
    }

    // Sort within each dim by seq ascending — deterministic regardless of absolute seq values
    for (auto& [dim, evs] : dim_events)
        std::sort(evs.begin(), evs.end(),
                  [](const JROMEvent* a, const JROMEvent* b){ return a->seq < b->seq; });

    // Build dim_order: anchor first, then remaining dims in canonical order
    std::vector<std::string> dim_order;
    if (!request.anchor.empty()) dim_order.push_back(request.anchor);
    for (const auto& cd : book_.canonical_dims)
        if (allowed.count(cd) && cd != request.anchor)
            dim_order.push_back(cd);

    res.anchor_satisfied = request.anchor.empty()
                         || !dim_events[request.anchor].empty();

    // serialize_view uses verbatim payload_json strings — never re-parses them —
    // so hash determinism is not affected by unordered_map key ordering in the payload.
    res.text            = serialize_view(dim_order, dim_events);
    res.projection_hash = util::sha256_hex(res.text);
    res.ok              = true;
    return res;
}

std::string ProjectionEngine::serialize_view(
    const std::vector<std::string>&                                        dim_order,
    const std::unordered_map<std::string, std::vector<const JROMEvent*>>& dim_events)
{
    std::string out;
    bool first_dim = true;
    for (const auto& dim : dim_order) {
        auto it = dim_events.find(dim);
        if (it == dim_events.end() || it->second.empty()) continue;

        if (!first_dim) out += '\n';
        first_dim = false;

        out += dim + '\n';
        for (const JROMEvent* ev : it->second)
            out += ev->type + ": " + ev->payload_json + '\n';
    }
    return out;
}

} // namespace khanary::io
