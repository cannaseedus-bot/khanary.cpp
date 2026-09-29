#include "sandbox_inspector.h"
#include <algorithm>
#include <vector>

namespace khanary::micronaut {

using namespace khanary::util;

void SandboxInspector::register_track(const std::string& id, TrackSandbox sb) {
    if (id.empty()) return;
    std::lock_guard<std::mutex> lk(mtx_);
    std::sort(sb.manifest_access.begin(), sb.manifest_access.end());
    sb.manifest_access.erase(std::unique(sb.manifest_access.begin(), sb.manifest_access.end()),
                             sb.manifest_access.end());
    std::sort(sb.effects.begin(), sb.effects.end());
    sb.effects.erase(std::unique(sb.effects.begin(), sb.effects.end()), sb.effects.end());
    std::sort(sb.ports.begin(), sb.ports.end());
    sb.ports.erase(std::unique(sb.ports.begin(), sb.ports.end()), sb.ports.end());
    sandboxes_[id] = std::move(sb);
}

bool SandboxInspector::has(const std::string& track_id) const {
    std::lock_guard<std::mutex> lk(mtx_);
    return sandboxes_.count(track_id) > 0;
}

bool SandboxInspector::check_effect(const std::string& track_id, const std::string& eff) {
    std::lock_guard<std::mutex> lk(mtx_);
    auto it = sandboxes_.find(track_id);
    if (it == sandboxes_.end()) return true;  // unregistered → open sandbox
    if (it->second.allows_effect(eff)) return true;
    record_violation(track_id, "effect", eff);
    return false;
}

bool SandboxInspector::check_port(const std::string& track_id, uint16_t port) {
    std::lock_guard<std::mutex> lk(mtx_);
    auto it = sandboxes_.find(track_id);
    if (it == sandboxes_.end()) return true;
    if (it->second.allows_port(port)) return true;
    record_violation(track_id, "port", std::to_string(port));
    return false;
}

bool SandboxInspector::check_manifest(const std::string& track_id, const std::string& manifest_name) {
    std::lock_guard<std::mutex> lk(mtx_);
    auto it = sandboxes_.find(track_id);
    if (it == sandboxes_.end()) return true;
    if (it->second.allows_manifest(manifest_name)) return true;
    record_violation(track_id, "manifest", manifest_name);
    return false;
}

std::string SandboxInspector::ns_key(const std::string& track_id, const std::string& raw_key) const {
    std::lock_guard<std::mutex> lk(mtx_);
    auto it = sandboxes_.find(track_id);
    if (it == sandboxes_.end()) return raw_key;
    return it->second.ns_key(raw_key);
}

void SandboxInspector::record_violation(const std::string& track_id,
                                         const std::string& kind,
                                         const std::string& detail) {
    violations_.push_back({track_id, kind, detail, seq_++});
}

// ── JSON serialisation ──────────────────────────────────────────────────────

JsonPtr SandboxInspector::sandbox_to_json(const std::string& id, const TrackSandbox& sb) {
    JsonObject obj;
    obj["id"]          = json_string(id);
    obj["gram_ns"]     = json_string(sb.gram_ns);
    obj["coordinator"] = json_bool(sb.coordinator);

    JsonArray ma;
    for (const auto& m : sb.manifest_access) ma.push_back(json_string(m));
    obj["manifest_access"] = json_array(std::move(ma));

    JsonArray ef;
    for (const auto& e : sb.effects) ef.push_back(json_string(e));
    obj["effects"] = json_array(std::move(ef));

    JsonArray ports;
    for (uint16_t p : sb.ports) ports.push_back(json_number((double)p));
    obj["ports"] = json_array(std::move(ports));

    return json_object(std::move(obj));
}

JsonPtr SandboxInspector::sandbox_json(const std::string& track_id) const {
    std::lock_guard<std::mutex> lk(mtx_);
    auto it = sandboxes_.find(track_id);
    if (it == sandboxes_.end()) return json_null();
    return sandbox_to_json(it->first, it->second);
}

JsonPtr SandboxInspector::all_json() const {
    std::lock_guard<std::mutex> lk(mtx_);
    JsonObject root;
    std::vector<std::string> ids;
    ids.reserve(sandboxes_.size());
    for (const auto& [id, _] : sandboxes_) ids.push_back(id);
    std::sort(ids.begin(), ids.end());
    for (const auto& id : ids) {
        const auto it = sandboxes_.find(id);
        if (it != sandboxes_.end())
            root[id] = sandbox_to_json(id, it->second);
    }
    return json_object(std::move(root));
}

JsonPtr SandboxInspector::violations_json(size_t n) const {
    std::lock_guard<std::mutex> lk(mtx_);
    JsonArray arr;
    size_t start = (n > 0 && n < violations_.size()) ? violations_.size() - n : 0;
    for (size_t i = start; i < violations_.size(); ++i) {
        const auto& v = violations_[i];
        JsonObject obj;
        obj["seq"]      = json_number((double)v.seq);
        obj["track_id"] = json_string(v.track_id);
        obj["kind"]     = json_string(v.kind);
        obj["detail"]   = json_string(v.detail);
        arr.push_back(json_object(std::move(obj)));
    }
    return json_array(std::move(arr));
}

} // namespace khanary::micronaut
