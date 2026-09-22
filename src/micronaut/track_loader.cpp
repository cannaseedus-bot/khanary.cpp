#include "track_loader.h"
#include "../util/json.h"
#include "../kuhul/phase_engine.h"
#include <fstream>
#include <sstream>
#include <filesystem>

namespace khanary::micronaut {

namespace fs = std::filesystem;
using namespace khanary::util;

TrackLoader::TrackLoader(MicronautRegistry& reg) : reg_(reg) {}

bool TrackLoader::load_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) return false;
    std::ostringstream ss; ss << f.rdbuf();
    auto root = parse(ss.str());
    if (!root || !root->is_object()) return false;

    MicronautEntry entry;
    entry.active     = true;
    entry.confidence = 1.0f;

    if (auto id = root->get_string("id")) entry.id = *id;
    if (auto desc = root->get_string("description")) entry.description = *desc;

    // Phase
    if (auto ph_s = root->get_string("phase")) {
        kuhul::Phase p;
        if (kuhul::phase_from_name(*ph_s, p)) entry.primary_phase = p;
        else entry.primary_phase = kuhul::Phase::Pop;
    } else {
        entry.primary_phase = kuhul::Phase::Pop;
    }

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

    // Delegation targets (Coordinator only, but parse for any track)
    const auto* sm = root->get("sub_micronauts");
    if (sm && sm->is_array())
        for (const auto& v : sm->as_array())
            if (v->is_string()) entry.sub_micronauts.push_back(v->as_string());

    // Confidence override
    if (auto conf = root->get_number("confidence")) entry.confidence = (float)*conf;

    if (entry.id.empty())
        entry.id = fs::path(path).stem().string();

    // Sandbox declaration (optional — defaults to open_sandbox() if absent)
    const auto* sb_node = root->get("sandbox");
    if (sb_node && sb_node->is_object()) {
        TrackSandbox sb;
        if (auto ns = sb_node->get_string("gram_ns"))    sb.gram_ns     = *ns;
        if (auto co = sb_node->get_bool("coordinator"))  sb.coordinator = *co;
        else sb.coordinator = false;

        auto parse_str_array = [&](const char* key, std::vector<std::string>& out) {
            const auto* arr = sb_node->get(key);
            if (arr && arr->is_array())
                for (const auto& v : arr->as_array())
                    if (v->is_string()) out.push_back(v->as_string());
        };
        parse_str_array("manifest_access", sb.manifest_access);
        parse_str_array("effects",         sb.effects);

        const auto* ports_node = sb_node->get("ports");
        if (ports_node && ports_node->is_array())
            for (const auto& pv : ports_node->as_array())
                if (pv->is_number()) sb.ports.push_back((uint16_t)(int)pv->as_number());

        entry.sandbox = std::move(sb);
    } else {
        entry.sandbox = open_sandbox();
    }

    reg_.add(std::move(entry));
    return true;
}

int TrackLoader::load_dir(const std::string& dir) {
    int count = 0;
    if (!fs::is_directory(dir)) return 0;
    for (const auto& de : fs::directory_iterator(dir)) {
        std::string fn = de.path().filename().string();
        if (fn.find(".semantic-tracks.v1.json") != std::string::npos)
            if (load_file(de.path().string())) ++count;
    }
    return count;
}

} // namespace khanary::micronaut
