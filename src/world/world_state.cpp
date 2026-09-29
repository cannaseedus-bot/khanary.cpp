#include "world_state.h"
#include <cmath>
#include <unordered_map>

namespace khanary::world {

using namespace khanary::util;

static constexpr double PI = 3.14159265358979323846;

// ── geometry helpers ───────────────────────────────────────────────────────────

float WorldState::pos_x(kuhul::Phase p, float r) {
    double angle = (2.0 * PI / 6.0) * static_cast<int>(p);
    return static_cast<float>(r * std::cos(angle));
}

float WorldState::pos_y(kuhul::Phase p, float r) {
    double angle = (2.0 * PI / 6.0) * static_cast<int>(p);
    return static_cast<float>(r * std::sin(angle));
}

// ── construction ───────────────────────────────────────────────────────────────

WorldState::WorldState(micronaut::MicronautRegistry& reg,
                       micronaut::SandboxInspector& inspector)
    : reg_(reg), inspector_(inspector)
{}

// ── rebuild ───────────────────────────────────────────────────────────────────

void WorldState::rebuild() {
    nodes_.clear();

    // Track how many nodes share each phase so we can z-offset them
    std::unordered_map<int, int> phase_count;

    // Get violation track ids from the inspector for O(n) lookup
    std::unordered_map<std::string, bool> has_violation;
    {
        auto v_arr = inspector_.violations_json(0);
        if (v_arr && v_arr->is_array()) {
            for (const auto& item : v_arr->as_array()) {
                if (!item) continue;
                auto tid = item->get_string("track_id");
                if (tid) has_violation[*tid] = true;
            }
        }
    }

    for (const auto& id : reg_.ids()) {
        const auto* e = reg_.get(id);
        if (!e) continue;

        auto phase = e->primary_phase;
        int  pi    = static_cast<int>(phase);

        int z_slot = phase_count[pi]++;

        TrackWorldNode node;
        node.track_id         = id;
        if (e->vector_meta.declared) {
            node.x = e->vector_meta.x;
            node.y = e->vector_meta.y;
            node.z = e->vector_meta.z;
            node.vector_source = e->vector_meta.source;
        } else {
            node.x = pos_x(phase);
            node.y = pos_y(phase);
            node.z = z_slot * 0.5f;
            node.vector_source = "phase_default";
        }
        node.primary_phase    = phase;
        node.active_block     = phase_to_block(phase);
        node.confidence       = e->confidence;
        node.static_weight    = e->static_weight;
        node.health           = e->health;
        node.gram_count       = 0;  // wired when GramStore exposes count_ns()
        node.sandbox_violation = has_violation.count(id) > 0;

        node.block_intensities.fill(0.0f);
        auto ab_idx = static_cast<size_t>(node.active_block);
        if (ab_idx < node.block_intensities.size())
            node.block_intensities[ab_idx] = 1.0f;

        nodes_.push_back(std::move(node));
    }
}

// ── JSON output ───────────────────────────────────────────────────────────────

JsonPtr WorldState::faces_json() {
    JsonArray arr;
    for (const auto& face : KCUBE_FACES) {
        JsonObject obj;
        obj["geometry"]  = json_string(std::string(face.geometry));
        obj["phase"]     = json_string(kuhul::phase_name(face.phase));
        obj["block"]     = json_string(std::string(face.block));
        obj["angle_deg"] = json_number(static_cast<double>(face.angle_deg));
        if (!face.extra_block.empty())
            obj["extra_block"] = json_string(std::string(face.extra_block));
        arr.push_back(json_object(std::move(obj)));
    }
    return json_array(std::move(arr));
}

JsonPtr WorldState::to_json() const {
    JsonArray tracks_arr;
    for (const auto& node : nodes_) {
        JsonObject obj;
        obj["id"]                = json_string(node.track_id);
        obj["phase"]             = json_string(kuhul::phase_name(node.primary_phase));

        // face name from KCUBE_FACES
        auto fi = static_cast<size_t>(node.primary_phase);
        if (fi < KCUBE_FACES.size())
            obj["face"] = json_string(std::string(KCUBE_FACES[fi].geometry));

        JsonArray pos;
        pos.push_back(json_number(static_cast<double>(node.x)));
        pos.push_back(json_number(static_cast<double>(node.y)));
        pos.push_back(json_number(static_cast<double>(node.z)));
        obj["pos"]               = json_array(std::move(pos));

        obj["active_block"]      = json_string(std::string(block_name(node.active_block)));
        obj["confidence"]        = json_number(static_cast<double>(node.confidence));
        obj["static_weight"]     = json_number(static_cast<double>(node.static_weight));
        obj["health"]            = json_number(static_cast<double>(node.health));
        obj["gram_count"]        = json_number(static_cast<double>(node.gram_count));
        obj["sandbox_violation"] = json_bool(node.sandbox_violation);
        obj["vector_source"]     = json_string(node.vector_source);

        JsonObject blocks_obj;
        for (uint8_t i = 0; i < static_cast<uint8_t>(AtomicDOMBlock::COUNT); ++i) {
            auto b = static_cast<AtomicDOMBlock>(i);
            blocks_obj[std::string(block_name(b))] =
                json_number(static_cast<double>(node.block_intensities[i]));
        }
        obj["blocks"] = json_object(std::move(blocks_obj));

        tracks_arr.push_back(json_object(std::move(obj)));
    }

    JsonObject root;
    root["faces"]  = faces_json();
    root["tracks"] = json_array(std::move(tracks_arr));
    return json_object(std::move(root));
}

} // namespace khanary::world
