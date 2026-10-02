#include "micronaut_registry.h"
#include "relation_isa.h"
#include <algorithm>
#include <cctype>
#include <cmath>

namespace khanary::micronaut {

namespace {

static float clamp01(float value, float fallback = 1.0f) {
    if (!std::isfinite(value)) return fallback;
    return std::clamp(value, 0.0f, 1.0f);
}

static std::vector<std::string> dedupe_non_empty(const std::vector<std::string>& in) {
    std::vector<std::string> out;
    out.reserve(in.size());
    std::unordered_set<std::string> seen;
    for (const auto& item : in) {
        if (item.empty()) continue;
        if (seen.insert(item).second) out.push_back(item);
    }
    return out;
}

static std::vector<std::string> dedupe_non_empty_lower(const std::vector<std::string>& in) {
    std::vector<std::string> out;
    out.reserve(in.size());
    std::unordered_set<std::string> seen;
    for (auto item : in) {
        if (item.empty()) continue;
        std::transform(item.begin(), item.end(), item.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (seen.insert(item).second) out.push_back(item);
    }
    return out;
}

static void sanitize_entry(MicronautEntry& entry) {
    entry.static_weight = clamp01(entry.static_weight, 1.0f);
    entry.health = clamp01(entry.health, 1.0f);
    entry.confidence = clamp01(entry.confidence, 1.0f);
    entry.capabilities = dedupe_non_empty(entry.capabilities);
    entry.sub_micronauts = dedupe_non_empty(entry.sub_micronauts);
    entry.route_contract.schemes = dedupe_non_empty_lower(entry.route_contract.schemes);
    entry.route_contract.verbs = dedupe_non_empty_lower(entry.route_contract.verbs);
    entry.route_contract.aiml_patterns = dedupe_non_empty(entry.route_contract.aiml_patterns);
    entry.route_contract.aiml_templates = dedupe_non_empty(entry.route_contract.aiml_templates);
    entry.route_contract.key_patterns = dedupe_non_empty(entry.route_contract.key_patterns);
    entry.route_contract.key_templates = dedupe_non_empty(entry.route_contract.key_templates);
    entry.route_contract.key_grammar = dedupe_non_empty(entry.route_contract.key_grammar);
    entry.route_contract.tokens = dedupe_non_empty_lower(entry.route_contract.tokens);
    if (entry.profile_projection && !entry.profile_projection->is_object())
        entry.profile_projection.reset();

    std::unordered_set<std::string> clean_governs;
    clean_governs.reserve(entry.governs.size());
    for (const auto& g : entry.governs) {
        if (!g.empty()) clean_governs.insert(g);
    }
    entry.governs.swap(clean_governs);

    std::vector<RelationEntry> clean_relations;
    clean_relations.reserve(entry.relations.size());
    for (const auto& rel : entry.relations) {
        if (rel.target_id.empty() || rel.verb.empty()) continue;
        clean_relations.push_back(rel);
    }
    entry.relations.swap(clean_relations);

    std::vector<FoldSpec> clean_folds;
    clean_folds.reserve(entry.folds.size());
    for (auto fold : entry.folds) {
        if (fold.op.empty()) continue;
        fold.weight = clamp01(fold.weight, 1.0f);
        clean_folds.push_back(std::move(fold));
    }
    entry.folds.swap(clean_folds);

    if (!std::isfinite(entry.vector_meta.x)) entry.vector_meta.x = 0.0f;
    if (!std::isfinite(entry.vector_meta.y)) entry.vector_meta.y = 0.0f;
    if (!std::isfinite(entry.vector_meta.z)) entry.vector_meta.z = 0.0f;
    if (entry.vector_meta.source.empty()) entry.vector_meta.source = "phase_default";
}

} // namespace

// Relation -> dominant K'uhul phase.
// Semantic neighborhood relations stay close to Pop/Yax;
// runtime/control/dependency relations escalate toward Sek/Chen/Xul.
kuhul::Phase verb_to_phase(const std::string& verb) {
    const auto rel = canonical_relation(verb);
    const auto family = relation_family_of(rel);

    if (rel == "@related_to") return kuhul::Phase::Pop;
    if (rel == "@is_a" || rel == "@part_of" || rel == "@has_a") return kuhul::Phase::Wo;
    if (rel == "@supports" || rel == "@contradicts" || rel == "@evidence_for")
        return kuhul::Phase::Chen;
    if (rel == "@delegates_to" || rel == "@return_to" || rel == "@resumes")
        return kuhul::Phase::Sek;
    if (rel == "@governs" || rel == "@redirect")
        return kuhul::Phase::Sek;

    switch (family) {
        case RelationFamily::Semantic:   return kuhul::Phase::Yax;
        case RelationFamily::Causal:     return kuhul::Phase::Yax;
        case RelationFamily::Epistemic:  return kuhul::Phase::Chen;
        case RelationFamily::Dependency: return kuhul::Phase::Sek;
        case RelationFamily::Context:    return kuhul::Phase::Wo;
        case RelationFamily::Control:    return kuhul::Phase::Sek;
        case RelationFamily::Runtime:    return kuhul::Phase::Sek;
        default:                         break;
    }
    return kuhul::Phase::Pop;
}

MicronautRegistry::MicronautRegistry() {}

void MicronautRegistry::add(const MicronautEntry& entry) {
    if (entry.id.empty()) return;
    MicronautEntry sanitized = entry;
    sanitize_entry(sanitized);

    std::lock_guard lk(mtx_);
    entries_[sanitized.id] = sanitized;
    gravity_.register_nodes(
        sanitized.id,
        sanitized.governs,
        sanitized.primary_phase,
        sanitized.vector_meta.x,
        sanitized.vector_meta.y,
        sanitized.vector_meta.z,
        sanitized.vector_meta.declared,
        sanitized.static_weight,
        sanitized.confidence,
        sanitized.health,
        0.0f,
        true,
        !sanitized.capabilities.empty() || sanitized.sandbox.coordinator,
        true);
}

void MicronautRegistry::add(MicronautEntry&& entry) {
    if (entry.id.empty()) return;
    sanitize_entry(entry);

    std::lock_guard lk(mtx_);
    std::string id = entry.id;
    auto governs = entry.governs;
    auto phase = entry.primary_phase;
    auto vmeta = entry.vector_meta;
    const float static_weight = entry.static_weight;
    const float confidence = entry.confidence;
    const float health = entry.health;
    const bool capable = !entry.capabilities.empty() || entry.sandbox.coordinator;
    entries_[id] = std::move(entry);
    gravity_.register_nodes(
        id, governs, phase, vmeta.x, vmeta.y, vmeta.z, vmeta.declared,
        static_weight, confidence, health, 0.0f, true, capable, true);
}

bool MicronautRegistry::remove(const std::string& id, float confidence_floor) {
    if (id.empty()) return false;
    const float floor = clamp01(confidence_floor, 0.7f);

    std::lock_guard lk(mtx_);
    auto it = entries_.find(id);
    if (it == entries_.end()) return false;
    if (it->second.confidence < floor) return false;
    entries_.erase(it);
    gravity_.unregister(id);
    return true;
}

bool MicronautRegistry::has(const std::string& id) const {
    if (id.empty()) return false;
    std::lock_guard lk(mtx_);
    return entries_.count(id) > 0;
}

const MicronautEntry* MicronautRegistry::get(const std::string& id) const {
    if (id.empty()) return nullptr;
    std::lock_guard lk(mtx_);
    auto it = entries_.find(id);
    if (it == entries_.end()) return nullptr;
    return &it->second;
}

MicronautEntry* MicronautRegistry::get_mut(const std::string& id) {
    if (id.empty()) return nullptr;
    std::lock_guard lk(mtx_);
    auto it = entries_.find(id);
    if (it == entries_.end()) return nullptr;
    return &it->second;
}

std::string MicronautRegistry::dispatch(
    kuhul::Phase phase,
    const std::vector<std::string>& tokens,
    float gravity_threshold) const
{
    if (tokens.empty()) return "";

    // Score by gravity
    auto scores = gravity_.score(tokens, phase);

    // Filter to active micronauts and pick deterministically.
    std::string best_id;
    const float threshold = std::isfinite(gravity_threshold) ? std::max(0.0f, gravity_threshold) : 0.1f;
    float best_score = threshold;
    constexpr float kEps = 1e-6f;
    {
        std::lock_guard lk(mtx_);
        for (const auto& [id, s] : scores) {
            auto it = entries_.find(id);
            if (it == entries_.end() || !it->second.active) continue;
            if (s > best_score + kEps ||
                (std::fabs(s - best_score) <= kEps && !best_id.empty() && id < best_id)) {
                best_score = s;
                best_id = id;
            }
        }
    }
    return best_id;
}

std::vector<std::pair<std::string, float>> MicronautRegistry::dispatch_blend(
    kuhul::Phase phase,
    const std::vector<std::string>& tokens,
    float gravity_threshold,
    size_t top_k) const
{
    if (top_k == 0 || tokens.empty()) return {};

    auto scores = gravity_.score(tokens, phase);

    const float threshold = std::isfinite(gravity_threshold) ? std::max(0.0f, gravity_threshold) : 0.1f;
    std::vector<std::pair<float, std::string>> ranked;
    {
        std::lock_guard lk(mtx_);
        for (const auto& [id, s] : scores) {
            auto it = entries_.find(id);
            if (it != entries_.end() && it->second.active && s >= threshold)
                ranked.push_back({s, id});
        }
    }
    if (ranked.empty()) return {};

    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
        if (a.first != b.first) return a.first > b.first;
        return a.second < b.second;
    });
    if (ranked.size() > top_k) ranked.resize(top_k);
    if (ranked.empty()) return {};

    // Normalize: weights sum to 1.0 (LBS invariant: Σw_k = 1)
    float total = 0.0f;
    for (const auto& [s, _] : ranked) total += s;
    const float inv = (total > 1e-6f) ? 1.0f / total : 1.0f / static_cast<float>(ranked.size());

    std::vector<std::pair<std::string, float>> blend;
    blend.reserve(ranked.size());
    for (const auto& [s, id] : ranked)
        blend.push_back({id, s * inv});
    return blend;
}

std::vector<std::string> MicronautRegistry::relational_dispatch(
    const std::string& from_id,
    const std::string& verb) const
{
    if (from_id.empty() || verb.empty()) return {};

    std::lock_guard lk(mtx_);
    auto it = entries_.find(from_id);
    if (it == entries_.end()) return {};
    std::vector<std::string> targets;
    std::unordered_set<std::string> seen;
    for (const auto& rel : it->second.relations) {
        auto target_it = entries_.find(rel.target_id);
        if (relation_matches(rel.verb, verb) && target_it != entries_.end() && target_it->second.active &&
            seen.insert(rel.target_id).second) {
            targets.push_back(rel.target_id);
        }
    }
    return targets;
}

std::vector<std::string> MicronautRegistry::ids() const {
    std::lock_guard lk(mtx_);
    std::vector<std::string> out;
    for (const auto& [id, _] : entries_) out.push_back(id);
    std::sort(out.begin(), out.end());
    return out;
}

} // namespace khanary::micronaut
