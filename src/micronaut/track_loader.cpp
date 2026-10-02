#include "track_loader.h"
#include "relation_isa.h"
#include "../util/json.h"
#include "../util/schema.h"
#include "../kuhul/phase_engine.h"
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <unordered_set>

namespace khanary::micronaut {

namespace fs = std::filesystem;
using namespace khanary::util;

namespace {

constexpr const char* kTrackSuffix = ".semantic-tracks.v1.json";

static bool has_track_suffix(const fs::path& p) {
    const std::string fn = p.filename().string();
    const size_t n = fn.size();
    const size_t m = std::char_traits<char>::length(kTrackSuffix);
    return n >= m && fn.compare(n - m, m, kTrackSuffix) == 0;
}

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

static void dedupe_ports(std::vector<uint16_t>& ports) {
    std::sort(ports.begin(), ports.end());
    ports.erase(std::unique(ports.begin(), ports.end()), ports.end());
}

static TrackVectorMeta default_vector_for_phase(kuhul::Phase phase) {
    constexpr double kPi = 3.14159265358979323846;
    const double angle = (2.0 * kPi / 6.0) * static_cast<int>(phase);
    TrackVectorMeta meta;
    meta.x = static_cast<float>(std::cos(angle));
    meta.y = static_cast<float>(std::sin(angle));
    meta.z = 0.0f;
    meta.declared = false;
    meta.source = "phase_default";
    return meta;
}

} // namespace

TrackLoader::TrackLoader(MicronautRegistry& reg) : reg_(reg) {}

bool TrackLoader::load_file(const std::string& path) {
    if (path.empty()) return false;
    const fs::path p = fs::path(path).lexically_normal();
    if (!has_track_suffix(p)) return false;
    if (!fs::exists(p) || !fs::is_regular_file(p)) return false;

    std::ifstream f(p);
    if (!f) return false;
    std::ostringstream ss; ss << f.rdbuf();
    JsonPtr root;
    try {
        root = parse(ss.str());
        std::string error;
        const auto schema = p.parent_path().parent_path() / "schema/micronaut-track.schema.json";
        if (!validate_schema_file(*root, schema, &error)) {
            std::cerr << "WARN: track rejected: " << error << "\n";
            return false;
        }
    } catch (const std::exception& e) {
        std::cerr << "WARN: track rejected: " << e.what() << "\n";
        return false;
    }
    if (!root || !root->is_object()) return false;

    if (auto schema = root->get_string("schema")) {
        const std::string schema_l = *schema;
        if (!(schema_l == "v1" || schema_l == "semantic-tracks.v1" || schema_l == "semantic-tracks.v1.json"))
            return false;
    }

    MicronautEntry entry;
    entry.active     = true;
    entry.confidence = 1.0f;
    entry.static_weight = 1.0f;
    entry.health = 1.0f;
    entry.no_eval = true;

    const auto no_eval = root->get_bool("@no-eval");
    if (!no_eval || !(*no_eval)) {
        std::cerr << "WARN: track rejected (requires \"@no-eval\": true): "
                  << p.string() << "\n";
        return false;
    }

    if (auto id   = root->get_string("id"))          entry.id          = *id;
    if (auto desc = root->get_string("description")) entry.description = *desc;
    if (auto cat  = root->get_string("category"))    entry.category    = *cat;

    // Phase
    if (auto ph_s = root->get_string("phase")) {
        kuhul::Phase parsed_phase;
        if (kuhul::phase_from_name(*ph_s, parsed_phase)) entry.primary_phase = parsed_phase;
        else return false;
    } else {
        return false;
    }
    entry.vector_meta = default_vector_for_phase(entry.primary_phase);

    auto parse_vector_meta = [&](const util::JsonValue* vec_node, const std::string& source) {
        if (!vec_node || !vec_node->is_object()) return;
        const auto vx = vec_node->get_number("x");
        const auto vy = vec_node->get_number("y");
        const auto vz = vec_node->get_number("z");
        if (!vx || !vy || !vz) return;
        if (!std::isfinite(*vx) || !std::isfinite(*vy) || !std::isfinite(*vz)) return;
        entry.vector_meta.x = static_cast<float>(*vx);
        entry.vector_meta.y = static_cast<float>(*vy);
        entry.vector_meta.z = static_cast<float>(*vz);
        entry.vector_meta.declared = true;
        entry.vector_meta.source = source;
    };
    parse_vector_meta(root->get("vector"), "track.vector");
    parse_vector_meta(root->get("@vector"), "track.@vector");

    // Governs
    const auto* gov = root->get("governs");
    if (gov && gov->is_array()) {
        for (const auto& g : gov->as_array())
            if (g->is_string()) entry.governs.insert(g->as_string());
    }

    // Capabilities
    const auto* caps = root->get("capabilities");
    if (caps && caps->is_array()) {
        for (const auto& c : caps->as_array())
            if (c->is_string()) entry.capabilities.push_back(c->as_string());
    }

    auto parse_str_array = [&](const util::JsonValue* node,
                               const char* key,
                               std::vector<std::string>& out) {
        if (!node || !node->is_object()) return;
        const auto* arr = node->get(key);
        if (!arr || !arr->is_array()) return;
        for (const auto& v : arr->as_array())
            if (v && v->is_string()) out.push_back(v->as_string());
    };

    // Optional per-micronaut URI/scheme callable surface contract.
    // Accepts either top-level arrays or nested objects:
    //   route_contract / address / uri / callable.
    parse_str_array(root.get(), "schemes", entry.route_contract.schemes);
    parse_str_array(root.get(), "verbs", entry.route_contract.verbs);
    parse_str_array(root.get(), "aiml_patterns", entry.route_contract.aiml_patterns);
    parse_str_array(root.get(), "aiml_templates", entry.route_contract.aiml_templates);
    parse_str_array(root.get(), "key_patterns", entry.route_contract.key_patterns);
    parse_str_array(root.get(), "key_templates", entry.route_contract.key_templates);
    parse_str_array(root.get(), "key_grammar", entry.route_contract.key_grammar);
    parse_str_array(root.get(), "tokens", entry.route_contract.tokens);

    const std::array<const char*, 4> kContractNodes = {
        "route_contract",
        "address",
        "uri",
        "callable"
    };
    for (const char* key : kContractNodes) {
        const auto* node = root->get(key);
        if (!node || !node->is_object()) continue;
        parse_str_array(node, "schemes", entry.route_contract.schemes);
        parse_str_array(node, "verbs", entry.route_contract.verbs);
        parse_str_array(node, "aiml_patterns", entry.route_contract.aiml_patterns);
        parse_str_array(node, "aiml_templates", entry.route_contract.aiml_templates);
        parse_str_array(node, "key_patterns", entry.route_contract.key_patterns);
        parse_str_array(node, "key_templates", entry.route_contract.key_templates);
        parse_str_array(node, "key_grammar", entry.route_contract.key_grammar);
        parse_str_array(node, "tokens", entry.route_contract.tokens);
    }

    // Optional profile/brain/progression projection. Preserve declared values as
    // a non-authoritative projection; runtime authority remains schema + phase gates.
    {
        util::JsonObject profile;
        bool any = false;
        for (const char* key : {"brain_profile", "personality", "capability_grams", "progression"}) {
            const auto* node = root->get(key);
            if (!node || !node->is_object()) continue;
            try {
                profile[key] = util::parse(util::stringify(*node));
                any = true;
            } catch (...) {
                // Malformed profile projections are ignored rather than promoted.
            }
        }
        if (any) {
            profile["projection_only"] = util::json_bool(true);
            profile["authority_boundary"] = util::json_string(
                "profile_projection_may_bias_routing_not_truth_or_commit");
            entry.profile_projection = util::json_object(std::move(profile));
        }
    }

    // Plain delegation targets (legacy — no verb)
    const auto* sm = root->get("sub_micronauts");
    if (sm && sm->is_array())
        for (const auto& v : sm->as_array())
            if (v->is_string()) entry.sub_micronauts.push_back(v->as_string());

    // Fold capability declarations:
    // { phase, op, key?, scope?, delegate?, note?, gate?, weight? }
    const auto* folds_node = root->get("folds");
    if (folds_node && folds_node->is_array()) {
        for (const auto& fv : folds_node->as_array()) {
            if (!fv || !fv->is_object()) {
                std::cerr << "WARN: track rejected (invalid fold entry object): "
                          << p.string() << "\n";
                return false;
            }

            auto phase_s = fv->get_string("phase");
            auto op_s = fv->get_string("op");
            if (!phase_s || phase_s->empty() || !op_s || op_s->empty()) {
                std::cerr << "WARN: track rejected (fold requires phase + op): "
                          << p.string() << "\n";
                return false;
            }

            kuhul::Phase fold_phase{};
            if (!kuhul::phase_from_name(*phase_s, fold_phase)) {
                std::cerr << "WARN: track rejected (unknown fold phase '" << *phase_s
                          << "'): " << p.string() << "\n";
                return false;
            }

            FoldSpec fold;
            fold.phase = fold_phase;
            fold.op = *op_s;
            if (auto key = fv->get_string("key")) fold.key = *key;
            if (auto scope = fv->get_string("scope")) fold.scope = *scope;
            if (auto delegate = fv->get_string("delegate")) fold.delegate = *delegate;
            if (auto note = fv->get_string("note")) fold.note = *note;
            if (auto gate = fv->get_string("gate")) fold.gate = *gate;
            if (auto w = fv->get_number("weight"))
                fold.weight = std::clamp(static_cast<float>(*w), 0.0f, 1.0f);

            entry.folds.push_back(std::move(fold));
        }
    }

    // Typed relation edges: [{"target":"id","verb":"@supports","directed":true}, ...]
    // Closed Relation ISA is enforced here (compatibility aliases allowed).
    const auto* rels = root->get("relations");
    if (rels && rels->is_array()) {
        for (const auto& rv : rels->as_array()) {
            if (!rv->is_object()) continue;
            RelationEntry rel;
            if (auto tgt  = rv->get_string("target"))   rel.target_id = *tgt;
            if (auto verb = rv->get_string("verb")) {
                if (!is_supported_relation(*verb)) {
                    std::cerr << "WARN: track rejected (unsupported relation verb '"
                              << *verb << "'): " << p.string() << "\n";
                    return false;
                }
                rel.verb = canonical_relation(*verb);
            }
            if (auto dir  = rv->get_bool("directed"))    rel.directed  = *dir;
            if (!rel.target_id.empty() && !rel.verb.empty())
                entry.relations.push_back(std::move(rel));
        }
    }

    // Confidence override
    if (auto conf = root->get_number("confidence")) entry.confidence = (float)*conf;
    if (auto sw = root->get_number("static_weight"))
        entry.static_weight = std::clamp((float)*sw, 0.0f, 1.0f);
    if (auto hs = root->get_number("health"))
        entry.health = std::clamp((float)*hs, 0.0f, 1.0f);

    // Sandbox declaration (optional — defaults to open_sandbox() if absent)
    const auto* sb_node = root->get("sandbox");
    if (sb_node && sb_node->is_object()) {
        TrackSandbox sb;
        if (auto ns = sb_node->get_string("gram_ns"))    sb.gram_ns     = *ns;
        if (auto co = sb_node->get_bool("coordinator"))  sb.coordinator = *co;
        else sb.coordinator = false;

        auto parse_sb_str_array = [&](const char* key, std::vector<std::string>& out) {
            const auto* arr = sb_node->get(key);
            if (arr && arr->is_array())
                for (const auto& v : arr->as_array())
                    if (v->is_string()) out.push_back(v->as_string());
        };
        parse_sb_str_array("manifest_access", sb.manifest_access);
        parse_sb_str_array("effects",         sb.effects);

        const auto* ports_node = sb_node->get("ports");
        if (ports_node && ports_node->is_array())
            for (const auto& pv : ports_node->as_array()) {
                if (!pv || !pv->is_number()) continue;
                const double raw = pv->as_number();
                if (!std::isfinite(raw)) continue;
                const int pnum = static_cast<int>(raw);
                if (pnum >= 1 && pnum <= 65535)
                    sb.ports.push_back(static_cast<uint16_t>(pnum));
            }

        sb.manifest_access = dedupe_non_empty(sb.manifest_access);
        sb.effects = dedupe_non_empty(sb.effects);
        if (!sb.coordinator && sb.gram_ns.empty()) sb.gram_ns = entry.id;
        dedupe_ports(sb.ports);

        entry.sandbox = std::move(sb);
    } else {
        entry.sandbox = open_sandbox();
    }

    entry.capabilities = dedupe_non_empty(entry.capabilities);
    entry.sub_micronauts = dedupe_non_empty(entry.sub_micronauts);
    entry.confidence = clamp01(entry.confidence, 1.0f);
    entry.static_weight = clamp01(entry.static_weight, 1.0f);
    entry.health = clamp01(entry.health, 1.0f);

    if (entry.id.empty()) return false;

    reg_.add(std::move(entry));
    return true;
}

int TrackLoader::load_dir(const std::string& dir) {
    int count = 0;
    if (!fs::is_directory(dir)) return 0;
    std::vector<fs::path> files;
    for (const auto& de : fs::directory_iterator(dir)) {
        if (!de.is_regular_file()) continue;
        if (has_track_suffix(de.path())) files.push_back(de.path());
    }
    std::sort(files.begin(), files.end());
    for (const auto& f : files)
        if (load_file(f.string())) ++count;

    for (const auto& id : reg_.ids()) {
        const auto* entry = reg_.get(id);
        if (!entry) continue;
        for (const auto& rel : entry->relations) {
            if (rel.target_id.empty()) continue;
            if (!reg_.has(rel.target_id)) {
                std::cerr << "WARN: track '" << id
                          << "' relation target not found: " << rel.target_id
                          << " (verb: " << rel.verb << ")\n";
            }
        }
    }
    return count;
}

} // namespace khanary::micronaut
