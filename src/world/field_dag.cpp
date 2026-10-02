#include "field_dag.h"
#include "gml_parser.h"
#include "../micronaut/relation_isa.h"
#include "../kuhul/phase_engine.h"
#include <sstream>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <unordered_set>

namespace khanary::world {

uint64_t FieldDag::next_substrate_id_ = 0;

namespace {

std::string lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

bool contains_any(const std::string& haystack, std::initializer_list<const char*> needles) {
    for (const char* n : needles) {
        if (haystack.find(n) != std::string::npos) return true;
    }
    return false;
}

std::string gml_source_token(const std::string& path) {
    const auto base_pos = path.find_last_of("/\\");
    const size_t base_start = (base_pos == std::string::npos) ? 0 : base_pos + 1;
    const auto dot_pos = path.find_last_of('.');
    const size_t base_end = (dot_pos != std::string::npos && dot_pos > base_start)
        ? dot_pos
        : path.size();

    std::string token = path.substr(base_start, base_end - base_start);
    if (token.empty()) token = "graph";

    for (char& c : token) {
        const auto uc = static_cast<unsigned char>(c);
        if (std::isalnum(uc) || c == '-' || c == '_') {
            c = static_cast<char>(std::tolower(uc));
        } else {
            c = '_';
        }
    }
    return token;
}

void assign_phase_face_block(FieldNode& n, kuhul::Phase p) {
    n.phase_name = kuhul::phase_name(p);
    const auto fi = static_cast<size_t>(p);
    if (fi < KCUBE_FACES.size()) {
        n.face_name  = KCUBE_FACES[fi].geometry;
        n.block_name = KCUBE_FACES[fi].block;
    } else {
        n.face_name  = n.phase_name;
        n.block_name = "";
    }
}

std::string classify_ontology_bucket(const micronaut::MicronautEntry& e) {
    std::string blob = lower_copy(e.id + " " + e.category + " " + e.description);
    for (const auto& g : e.governs) {
        blob += ' ';
        blob += lower_copy(g);
    }

    if (contains_any(blob, {"coordinator", "orchestr", "mission", "objective", "goal", "plan", "executive"}))
        return "Organization";
    if (contains_any(blob, {"perspective", "view", "concern", "scenario", "profile"}))
        return "Perspective";
    if (contains_any(blob, {"reason", "infer", "logic", "thought", "theory", "mind", "conscious", "evaluate", "verify"}))
        return "Function";
    if (contains_any(blob, {"knowledge", "data", "metadata", "memory", "search", "research", "citation", "information", "retrieve", "librarian"}))
        return "Information";
    if (contains_any(blob, {"component", "app", "wizard", "factory", "code", "build", "template", "module", "service"}))
        return "Component";
    if (contains_any(blob, {"communic", "protocol", "message", "meeting", "email", "translate"}))
        return "Communication";
    if (contains_any(blob, {"connect", "router", "route", "dispatch", "bridge", "adapter", "tool-call", "connector"}))
        return "Connector";
    if (contains_any(blob, {"environment", "world", "runtime", "physical", "sandbox", "gpu"}))
        return "Environment";

    switch (e.primary_phase) {
        case kuhul::Phase::Pop:  return "Information";
        case kuhul::Phase::Wo:   return "Component";
        case kuhul::Phase::Yax:  return "Perspective";
        case kuhul::Phase::Sek:  return "Function";
        case kuhul::Phase::Chen: return "Organization";
        case kuhul::Phase::Xul:  return "Environment";
    }
    return "Information";
}

FieldEdge make_ontology_edge(const std::string& from,
                             const std::string& to,
                             const std::string& verb,
                             bool directed,
                             float weight = 1.0f,
                             float delta_theta = 0.0f) {
    FieldEdge e;
    e.from = from;
    e.to = to;
    e.type = "ontology";
    e.verb = micronaut::canonical_relation(verb);
    if (!micronaut::is_supported_relation(e.verb))
        e.verb = micronaut::infer_relation_from_lexical(verb);
    e.relation_phase = micronaut::relation_phase_hint(e.verb);
    e.directed = directed;
    e.weight = weight;
    e.delta_theta = delta_theta;
    e.xcfe_required = false;
    return e;
}

} // namespace

FieldDag::FieldDag(const WorldState& ws, const micronaut::MicronautRegistry& reg)
    : ws_(ws), reg_(reg)
{}

// ── build ─────────────────────────────────────────────────────────────────────

void FieldDag::build() {
    nodes_.clear();
    edges_.clear();

    // Index WorldState nodes by track_id for quick lookup
    std::unordered_map<std::string, const TrackWorldNode*> node_map;
    for (const auto& wn : ws_.nodes())
        node_map[wn.track_id] = &wn;

    // Phase index: phase → list of track_ids at that phase
    std::unordered_map<int, std::vector<std::string>> phase_to_tracks;
    // Track → ontology class link
    std::unordered_map<std::string, std::string> track_to_ontology;

    // Build FieldNodes from WorldState (which is already built from registry)
    for (const auto& id : reg_.ids()) {
        const auto* e = reg_.get(id);
        if (!e) continue;
        auto wn_it = node_map.find(id);
        if (wn_it == node_map.end()) continue;
        const auto* wn = wn_it->second;

        FieldNode fn;
        fn.track_id    = id;
        fn.phase_name  = kuhul::phase_name(e->primary_phase);
        fn.block_name  = std::string(block_name(wn->active_block));
        fn.confidence  = std::clamp(e->confidence, 0.0f, 1.0f);
        fn.static_weight = std::clamp(e->static_weight, 0.0f, 1.0f);
        fn.health = std::clamp(e->health, 0.0f, 1.0f);
        fn.wcr_prior = std::clamp(fn.static_weight * fn.confidence * fn.health, 0.0f, 1.0f);
        fn.x           = wn->x;
        fn.y           = wn->y;
        fn.z           = wn->z;
        fn.vector_source = e->vector_meta.declared ? e->vector_meta.source : "phase_default";
        fn.kind              = "track";
        fn.category          = e->category;
        fn.coordinator       = e->sandbox.coordinator;
        fn.effects           = e->sandbox.effects;
        fn.block_intensities = wn->block_intensities;

        auto fi = static_cast<size_t>(e->primary_phase);
        fn.face_name = (fi < KCUBE_FACES.size())
                           ? std::string(KCUBE_FACES[fi].geometry)
                           : fn.phase_name;

        phase_to_tracks[static_cast<int>(e->primary_phase)].push_back(id);
        track_to_ontology[id] = classify_ontology_bucket(*e);
        nodes_.push_back(std::move(fn));
    }

    // ── Edge set 1: relational verb edges + legacy delegate arcs ────────────
    for (const auto& id : reg_.ids()) {
        const auto* e = reg_.get(id);
        if (!e) continue;

        // Typed relational edges from the `relations` table
        for (const auto& rel : e->relations) {
            if (!node_map.count(rel.target_id)) continue;
            kuhul::Phase target_phase = kuhul::Phase::Pop;
            if (const auto* te = reg_.get(rel.target_id))
                target_phase = te->primary_phase;
            float dtheta = static_cast<float>(
                kuhul::phase_distance(e->primary_phase, target_phase)) * 60.0f;

            FieldEdge edge;
            edge.from          = id;
            edge.to            = rel.target_id;
            edge.verb          = micronaut::canonical_relation(rel.verb);
            if (!micronaut::is_supported_relation(edge.verb))
                edge.verb = micronaut::infer_relation_from_lexical(rel.verb);
            edge.relation_phase = micronaut::relation_phase_hint(edge.verb);
            edge.directed      = rel.directed;
            edge.type          = rel.directed ? "delegate" : "horizontal_fold";
            edge.weight        = rel.directed ? std::cos(dtheta * 3.14159f / 180.0f) : 1.0f;
            edge.delta_theta   = rel.directed ? dtheta : 0.0f;
            edge.xcfe_required = rel.directed && !micronaut::relation_matches(edge.verb, "@delegates_to");
            edges_.push_back(std::move(edge));
        }

        // Legacy plain-delegate fallback (no verb — used when relations is empty)
        if (e->relations.empty()) {
            for (const auto& target : e->sub_micronauts) {
                if (!node_map.count(target)) continue;
                FieldEdge edge;
                edge.from          = id;
                edge.to            = target;
                edge.type          = "delegate";
                edge.verb          = "@delegates_to";
                edge.relation_phase = micronaut::relation_phase_hint(edge.verb);
                edge.weight        = 1.0f;
                edge.delta_theta   = 0.0f;
                edge.xcfe_required = false;
                edges_.push_back(std::move(edge));
            }
        }
    }

    // ── Edge set 2: C₆ vertical fold arcs (adjacent phases) ─────────────────
    // Directed: phase_i → next_phase; arc weight = cos(π/3) = 0.5
    for (const auto& id_a : reg_.ids()) {
        const auto* ea = reg_.get(id_a);
        if (!ea) continue;
        kuhul::Phase next_p = kuhul::next_phase(ea->primary_phase);
        auto it = phase_to_tracks.find(static_cast<int>(next_p));
        if (it == phase_to_tracks.end()) continue;
        for (const auto& id_b : it->second) {
            FieldEdge edge;
            edge.from          = id_a;
            edge.to            = id_b;
            edge.type          = "vertical_fold";
            edge.weight        = 0.5f;  // cos(π/3)
            edge.delta_theta   = 60.0f;
            edge.xcfe_required = true;
            edges_.push_back(std::move(edge));
        }
    }

    // ── Edge set 3: horizontal fold arcs (same phase, Δθ=0, weight=1.0) ─────
    // Undirected pair emitted once (from < to lexicographically)
    for (const auto& [pi, ids] : phase_to_tracks) {
        if (ids.size() < 2) continue;
        for (size_t i = 0; i < ids.size(); ++i) {
            for (size_t j = i + 1; j < ids.size(); ++j) {
                FieldEdge edge;
                edge.from          = ids[i];
                edge.to            = ids[j];
                edge.type          = "horizontal_fold";
                edge.weight        = 1.0f;  // cos(0)
                edge.delta_theta   = 0.0f;
                edge.xcfe_required = false;
                edges_.push_back(edge);
                // Reverse direction (symmetric)
                std::swap(edge.from, edge.to);
                edges_.push_back(std::move(edge));
            }
        }
    }

    // ── Ontology overlay: MBED top-level classes + typed links ──────────────
    auto add_ontology_node =
        [&](const char* id,
            kuhul::Phase phase,
            float x, float y, float z,
            std::initializer_list<const char*> attrs) {
            FieldNode n;
            n.kind = "ontology";
            n.track_id = id;
            n.category = "ontology";
            n.confidence = 1.0f;
            n.static_weight = 1.0f;
            n.health = 1.0f;
            n.wcr_prior = 1.0f;
            n.x = x;
            n.y = y;
            n.z = z;
            n.vector_source = "ontology_static";
            n.coordinator = false;
            assign_phase_face_block(n, phase);
            for (const char* a : attrs) n.effects.push_back(a);
            nodes_.push_back(std::move(n));
        };

    add_ontology_node("Organization", kuhul::Phase::Chen,  0.05f,  1.85f, 0.0f,
                      {"mission", "requirements", "objectives", "goals", "scenarios"});
    add_ontology_node("Perspective",  kuhul::Phase::Yax,   2.10f,  1.45f, 0.0f,
                      {"contains_objects", "defines", "exposes_concerns"});
    add_ontology_node("Function",     kuhul::Phase::Sek,  -2.05f,  0.25f, 0.0f,
                      {"logical_structure", "behavior", "interfaces", "constraints"});
    add_ontology_node("Information",  kuhul::Phase::Pop,   2.05f,  0.20f, 0.0f,
                      {"data", "metadata", "rules"});
    add_ontology_node("Component",    kuhul::Phase::Wo,    0.10f, -0.95f, 0.0f,
                      {"type", "attributes", "ports", "location"});
    add_ontology_node("Environment",  kuhul::Phase::Xul,   2.00f, -1.00f, 0.0f,
                      {"physical_environs", "attributes"});
    add_ontology_node("Communication",kuhul::Phase::Yax,  -1.95f, -1.45f, 0.0f,
                      {"protocol_stack", "standards"});
    add_ontology_node("Connector",    kuhul::Phase::Wo,    1.20f, -1.85f, 0.0f,
                      {"type", "attributes"});

    std::unordered_set<std::string> ontology_ids = {
        "Organization", "Perspective", "Function", "Information",
        "Component", "Environment", "Communication", "Connector"
    };

    for (const auto& [track_id, onto_id] : track_to_ontology) {
        if (!ontology_ids.count(onto_id)) continue;
        edges_.push_back(make_ontology_edge(track_id, onto_id, "@related_to", false, 1.0f, 0.0f));
    }

    // Core ontology relationships (from MBED top-level pattern).
    edges_.push_back(make_ontology_edge("Organization", "Function", "@part_of", true, 0.85f, 30.0f));
    edges_.push_back(make_ontology_edge("Organization", "Component", "@delegates_to", true, 0.85f, 30.0f));
    edges_.push_back(make_ontology_edge("Perspective", "Organization", "@related_to", false, 1.0f, 0.0f));
    edges_.push_back(make_ontology_edge("Perspective", "Information", "@related_to", false, 1.0f, 0.0f));
    edges_.push_back(make_ontology_edge("Function", "Information", "@delegates_to", true, 0.8f, 30.0f));
    edges_.push_back(make_ontology_edge("Information", "Function", "@supports", true, 0.75f, 30.0f));
    edges_.push_back(make_ontology_edge("Function", "Component", "@delegates_to", true, 0.8f, 30.0f));
    edges_.push_back(make_ontology_edge("Communication", "Function", "@delegates_to", true, 0.8f, 30.0f));
    edges_.push_back(make_ontology_edge("Function", "Communication", "@part_of", true, 0.8f, 30.0f));
    edges_.push_back(make_ontology_edge("Communication", "Component", "@delegates_to", true, 0.8f, 30.0f));
    edges_.push_back(make_ontology_edge("Component", "Connector", "@delegates_to", true, 0.8f, 30.0f));
    edges_.push_back(make_ontology_edge("Connector", "Component", "@part_of", true, 0.8f, 30.0f));
    edges_.push_back(make_ontology_edge("Component", "Environment", "@governs", true, 0.8f, 30.0f));
}

// ── add_tensor ────────────────────────────────────────────────────────────────

void FieldDag::add_tensor(const std::string& id,
                          const std::string& /*shape*/,
                          std::vector<FoldEntry> entries)
{
    FieldNode fn;
    fn.kind       = "tensor";
    fn.track_id   = id;
    fn.confidence = 1.0f;
    fn.static_weight = 1.0f;
    fn.health = 1.0f;
    fn.wcr_prior = 1.0f;
    fn.vector_source = "tensor_projection";
    fn.fold_map   = std::move(entries);
    if (!fn.fold_map.empty()) {
        fn.phase_name = fn.fold_map[0].phase_name;
        kuhul::Phase ph = kuhul::Phase::Pop;
        kuhul::phase_from_name(fn.phase_name, ph);
        auto sp = phase_svg_pos(ph);
        fn.x = (sp.x - SVG_CX) / SVG_SCALE;
        fn.y = (SVG_CY - sp.y) / SVG_SCALE;
        fn.z = 0.0f;
        auto fi = static_cast<size_t>(ph);
        if (fi < KCUBE_FACES.size()) {
            fn.face_name  = std::string(KCUBE_FACES[fi].geometry);
            fn.block_name = std::string(KCUBE_FACES[fi].block);
        }
    }
    nodes_.push_back(std::move(fn));
}

// ── GML overlay ───────────────────────────────────────────────────────────────

void FieldDag::load_gml(const std::string& path) {
    auto g = parse_gml_file(path);
    if (!g) return;

    const std::string source_token = gml_source_token(path);
    const size_t N = g->nodes.size();
    constexpr double kPI = 3.14159265358979323846;

    // Build id → track_id map for edge resolution
    std::unordered_map<int64_t, std::string> id_to_tid;
    id_to_tid.reserve(N);

    for (size_t i = 0; i < N; ++i) {
        const auto& gn = g->nodes[i];

        // If the label names a registered track, bind to it directly —
        // no phantom node, edges will wire into the real track.
        if (!gn.label.empty() && reg_.get(gn.label)) {
            id_to_tid[gn.id] = gn.label;
            continue;
        }

        FieldNode fn;
        fn.kind        = "gml";
        fn.track_id    = "gml:" + source_token + ":" + std::to_string(gn.id);
        fn.phase_name  = "Pop";
        fn.confidence  = std::clamp(static_cast<float>(gn.confidence), 0.0f, 1.0f);
        fn.static_weight = std::clamp(static_cast<float>(gn.static_weight), 0.0f, 1.0f);
        fn.health = std::clamp(static_cast<float>(gn.health), 0.0f, 1.0f);
        fn.wcr_prior = std::clamp(fn.confidence * fn.static_weight * fn.health, 0.0f, 1.0f);
        fn.vector_source = "gml";
        fn.coordinator = false;
        if (!gn.label.empty()) fn.category = gn.label;

        // field geometry from GML overlay
        if (gn.has_local_entropy)    fn.local_entropy    = static_cast<float>(gn.local_entropy);
        if (gn.has_geodesic_entropy) fn.geodesic_entropy = static_cast<float>(gn.geodesic_entropy);
        if (gn.has_unruh_temp)       fn.unruh_temp       = static_cast<float>(gn.unruh_temp);
        if (gn.has_divergence)       fn.divergence       = static_cast<float>(gn.divergence);
        if (gn.has_circumradius_R)   fn.circumradius_R   = static_cast<float>(gn.circumradius_R);
        if (gn.has_inradius_r)       fn.inradius_r       = static_cast<float>(gn.inradius_r);
        if (!gn.singularity.empty()) {
            if      (gn.singularity == "source")  fn.singularity_type = 1u;
            else if (gn.singularity == "sink")    fn.singularity_type = 2u;
            else if (gn.singularity == "saddle")  fn.singularity_type = 3u;
            else                                  fn.singularity_type = 0u;
        }

        if (!gn.provider_id.empty()) {
            fn.provider_id  = gn.provider_id;
            fn.parameter_id = gn.parameter_id;

            // Upsert a projection-only TensorSubstrate per unique provider identity.
            // Does NOT create a MicronautEntry — geometry source, not semantic authority.
            if (substrates_.find(gn.provider_id) == substrates_.end()) {
                TensorSubstrate ts;
                ts.tensor_id         = next_substrate_id_++;
                ts.provider_identity = gn.provider_id;
                ts.dtype             = DataType::FLOAT32;
                ts.residency         = DeviceResidency::HOST_RAM;
                substrates_[gn.provider_id] = std::move(ts);
            }
        }

        if (gn.has_x && gn.has_y) {
            fn.x = static_cast<float>(gn.x);
            fn.y = static_cast<float>(gn.y);
            fn.z = gn.has_z ? static_cast<float>(gn.z) : 0.0f;
        } else {
            double angle = (N > 1)
                ? (2.0 * kPI * static_cast<double>(i) / static_cast<double>(N))
                : 0.0;
            fn.x = static_cast<float>(2.5 * std::cos(angle));
            fn.y = static_cast<float>(2.5 * std::sin(angle));
            fn.z = 0.0f;
        }

        id_to_tid[gn.id] = fn.track_id;
        nodes_.push_back(std::move(fn));
    }

    for (const auto& ge : g->edges) {
        auto sit = id_to_tid.find(ge.source);
        auto tit = id_to_tid.find(ge.target);
        if (sit == id_to_tid.end() || tit == id_to_tid.end()) continue;

        FieldEdge fe;
        fe.from          = sit->second;
        fe.to            = tit->second;
        fe.directed      = g->directed;
        fe.type          = g->directed ? "delegate" : "horizontal_fold";
        fe.weight        = ge.has_weight ? static_cast<float>(ge.weight) : 1.0f;
        fe.delta_theta   = 0.0f;
        fe.xcfe_required = false;
        // field geometry
        if (ge.has_length) fe.length = static_cast<float>(ge.length);
        if (ge.has_vx)     fe.vx     = static_cast<float>(ge.vx);
        if (ge.has_vy)     fe.vy     = static_cast<float>(ge.vy);
        if (ge.has_vz)     fe.vz     = static_cast<float>(ge.vz);
        if (ge.has_eh_x)   fe.eh_x   = static_cast<float>(ge.eh_x);
        if (ge.has_eh_y)   fe.eh_y   = static_cast<float>(ge.eh_y);
        if (ge.has_eh_z)   fe.eh_z   = static_cast<float>(ge.eh_z);

        const std::string relation_raw = !ge.verb.empty() ? ge.verb : ge.label;
        if (!relation_raw.empty()) {
            std::string typed = relation_raw;
            if (!typed.empty() && typed.front() == '@') {
                typed = micronaut::canonical_relation(typed);
            } else {
                typed = micronaut::infer_relation_from_lexical(relation_raw);
            }
            if (!typed.empty()) {
                fe.verb = typed;
                fe.relation_phase = micronaut::relation_phase_hint(typed);
                fe.xcfe_required = g->directed && !micronaut::relation_matches(typed, "@delegates_to");
            }
        }
        edges_.push_back(std::move(fe));
    }
}

// ── XML serialization ─────────────────────────────────────────────────────────

std::string FieldDag::xml_escape(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        switch (c) {
            case '&':  out += "&amp;";  break;
            case '<':  out += "&lt;";   break;
            case '>':  out += "&gt;";   break;
            case '"':  out += "&quot;"; break;
            case '\'': out += "&apos;"; break;
            default:   out += c;
        }
    }
    return out;
}

std::string FieldDag::to_xml() const {
    std::ostringstream os;
    os << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
    os << "<field version=\"1.0\" phase_cycle=\"C6\">\n";

    // ── faces section (K-CUBE face table) ────────────────────────────────────
    os << "  <faces>\n";
    for (const auto& face : KCUBE_FACES) {
        os << "    <face"
           << " id=\""        << xml_escape(face.geometry)            << "\""
           << " phase=\""     << xml_escape(kuhul::phase_name(face.phase)) << "\""
           << " block=\""     << xml_escape(face.block)               << "\""
           << " angle_deg=\"" << static_cast<int>(face.angle_deg)     << "\"";
        if (!face.extra_block.empty())
            os << " extra_block=\"" << xml_escape(face.extra_block) << "\"";
        os << "/>\n";
    }
    os << "  </faces>\n";

    // ── nodes ─────────────────────────────────────────────────────────────────
    os << "  <nodes>\n";
    // XCFE phase anchor nodes — @phase: typed, one per C6 face
    for (const auto& face : KCUBE_FACES) {
        kuhul::Phase next_p = kuhul::next_phase(face.phase);
        char abuf[16];
        std::snprintf(abuf, sizeof(abuf), "%.3f", kuhul::phase_angle(face.phase));
        os << "    <node"
           << " id=\"@phase:"   << xml_escape(kuhul::phase_name(face.phase)) << "\""
           << " type=\"xcfe_phase\""
           << " face=\""        << xml_escape(face.geometry) << "\""
           << " block=\""       << xml_escape(face.block)    << "\""
           << " angle_deg=\""   << static_cast<int>(face.angle_deg) << "\""
           << " angle_rad=\""   << abuf << "\""
           << " next=\"@phase:" << xml_escape(kuhul::phase_name(next_p)) << "\""
           << "/>\n";
    }
    for (const auto& n : nodes_) {
        os << "    <node"
           << " id=\"@node:"    << xml_escape(n.track_id)   << "\""
           << " kind=\""        << xml_escape(n.kind)        << "\""
           << " phase=\""       << xml_escape(n.phase_name) << "\""
           << " face=\""        << xml_escape(n.face_name)  << "\""
           << " block=\""       << xml_escape(n.block_name) << "\""
           << " confidence=\""  << n.confidence             << "\""
           << " static_weight=\"" << n.static_weight        << "\""
           << " health=\""      << n.health                 << "\""
           << " wcr_prior=\""   << n.wcr_prior              << "\""
           << " coordinator=\"" << (n.coordinator ? "true" : "false") << "\"";
        if (!n.category.empty())
           os << " category=\"" << xml_escape(n.category) << "\"";
        os << " vector_source=\"" << xml_escape(n.vector_source) << "\"";
        os << " x=\""           << n.x << "\""
           << " y=\""           << n.y << "\""
           << " z=\""           << n.z << "\"";
        os << ">\n";
        if (!n.effects.empty()) {
            os << "      <effects>";
            for (size_t i = 0; i < n.effects.size(); ++i) {
                if (i) os << ' ';
                os << xml_escape(n.effects[i]);
            }
            os << "</effects>\n";
        }
        os << "      <vector_meta"
           << " source=\"" << xml_escape(n.vector_source) << "\""
           << " qkv_role=\"K\""
           << " x=\"" << n.x << "\""
           << " y=\"" << n.y << "\""
           << " z=\"" << n.z << "\""
           << "/>\n";
        // SVG-3D weight layer: one float per AtomicDOMBlock
        os << "      <tensors";
        for (uint8_t bi = 0; bi < static_cast<uint8_t>(AtomicDOMBlock::COUNT); ++bi) {
            auto b = static_cast<AtomicDOMBlock>(bi);
            os << " " << block_name(b) << "=\"" << n.block_intensities[bi] << "\"";
        }
        os << "/>\n";
        // fold_map: multi-phase presence declaration (tensors + pipeline stages)
        if (!n.fold_map.empty()) {
            os << "      <fold_map>\n";
            for (const auto& fe : n.fold_map) {
                os << "        <entry phase=\"" << xml_escape(fe.phase_name) << "\"";
                if (!fe.hook.empty())    os << " hook=\""    << xml_escape(fe.hook)    << "\"";
                if (!fe.monitor.empty()) os << " monitor=\"" << xml_escape(fe.monitor) << "\"";
                if (!fe.action.empty())  os << " action=\""  << xml_escape(fe.action)  << "\"";
                os << "/>\n";
            }
            os << "      </fold_map>\n";
        }
        os << "    </node>\n";
    }
    os << "  </nodes>\n";

    // ── edges ─────────────────────────────────────────────────────────────────
    os << "  <edges>\n";
    for (const auto& e : edges_) {
        os << "    <edge"
           << " from=\"@node:"    << xml_escape(e.from)  << "\""
           << " to=\"@node:"      << xml_escape(e.to)    << "\""
           << " type=\""          << xml_escape(e.type)  << "\"";
        if (!e.verb.empty())
           os << " verb=\""       << xml_escape(e.verb)  << "\"";
        if (!e.relation_phase.empty())
           os << " relation_phase=\"" << xml_escape(e.relation_phase) << "\"";
        os << " weight=\""        << e.weight             << "\""
           << " delta_theta=\""   << static_cast<int>(e.delta_theta) << "\""
           << " xcfe_required=\"" << (e.xcfe_required ? "true" : "false") << "\""
           << " directed=\""      << (e.directed ? "true" : "false") << "\""
           << "/>\n";
    }
    // Phase anchor edges: @node:X → @phase:P
    for (const auto& n : nodes_) {
        os << "    <edge"
           << " from=\"@node:" << xml_escape(n.track_id)   << "\""
           << " to=\"@phase:"  << xml_escape(n.phase_name) << "\""
           << " type=\"phase_anchor\" arc_weight=\"1.000\" directed=\"true\""
           << "/>\n";
    }
    os << "  </edges>\n";

    os << "</field>\n";
    return os.str();
}

// ── SVG rendering ─────────────────────────────────────────────────────────────

// Block colors (ASX palette)
static constexpr const char* BLOCK_COLORS[7] = {
    "#22f2b3",  // perceive  → --asx-primary
    "#38bdf8",  // learn     → --asx-secondary
    "#a855f7",  // think     → --asx-accent
    "#facc15",  // evaluate  → --asx-warning
    "#4ade80",  // confidence→ --asx-success
    "#fb7185",  // memory    → --asx-danger
    "#fbbf24",  // evolve    → --asx-hazard
};

// Relational verb colors (ASX palette)
static const char* verb_color(const std::string& verb) {
    const auto rel = micronaut::canonical_relation(verb);
    if (rel == "@delegates_to" || rel == "@return_to" || rel == "@resumes")
        return "#38bdf8"; // runtime
    if (rel == "@is_a" || rel == "@part_of" || rel == "@has_a")
        return "#22c55e"; // structural semantic
    if (rel == "@related_to")
        return "#22f2b3"; // weak semantic association
    if (rel == "@causes" || rel == "@enables" || rel == "@prevents")
        return "#fb7185"; // causal
    if (rel == "@supports" || rel == "@contradicts" || rel == "@evidence_for")
        return "#facc15"; // epistemic evidence
    if (rel == "@requires" || rel == "@produces" || rel == "@depends_on")
        return "#a855f7"; // dependency/process
    if (rel == "@governs" || rel == "@redirect")
        return "#eab308"; // control scope / redirect
    if (rel == "@context" || rel == "@previous" || rel == "@next")
        return "#14b8a6"; // contextual chain
    return nullptr;
}

// Edge type colors (fallback when no verb is set)
static const char* edge_color(const std::string& type, const std::string& verb) {
    if (const char* vc = verb_color(verb)) return vc;
    if (type == "ontology")        return "#fbbf24";
    if (type == "delegate")        return "#22f2b3";
    if (type == "horizontal_fold") return "#38bdf8";
    return "#6b7280";  // vertical_fold — faint
}

static const char* ontology_fill(const std::string& id) {
    if (id == "Organization")  return "#0891b2";
    if (id == "Perspective")   return "#0e7490";
    if (id == "Function")      return "#a21caf";
    if (id == "Information")   return "#dc2626";
    if (id == "Component")     return "#0284c7";
    if (id == "Environment")   return "#0369a1";
    if (id == "Communication") return "#ca8a04";
    if (id == "Connector")     return "#2563eb";
    return "#334155";
}

FieldDag::SVGPos FieldDag::phase_svg_pos(kuhul::Phase p) {
    constexpr double kPI = 3.14159265358979323846;
    double angle = (2.0 * kPI / 6.0) * static_cast<int>(p);
    return {
        SVG_CX + static_cast<float>(SVG_SCALE * std::cos(angle)),
        SVG_CY - static_cast<float>(SVG_SCALE * std::sin(angle))
    };
}

FieldDag::SVGPos FieldDag::to_svg_pos(const FieldNode& n) {
    // World coords: x,y in [-2,2]; z is stacking offset (0, 0.5, 1.0, ...)
    // SVG: y-axis inverted; z adds a diagonal offset so stacked tracks are visible
    return {
        SVG_CX + n.x * SVG_SCALE + n.z * 22.0f,
        SVG_CY - n.y * SVG_SCALE - n.z * 12.0f
    };
}

std::string FieldDag::svg_edge_path(const SVGPos& a, const SVGPos& b) {
    // Bezier with midpoint control pulled slightly inward toward center
    float mx = (a.x + b.x) * 0.5f + (SVG_CX - (a.x + b.x) * 0.5f) * 0.15f;
    float my = (a.y + b.y) * 0.5f + (SVG_CY - (a.y + b.y) * 0.5f) * 0.15f;
    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "M %.1f,%.1f Q %.1f,%.1f %.1f,%.1f",
        a.x, a.y, mx, my, b.x, b.y);
    return buf;
}

// Draw 7 isometric tensor bars above a node, colored by block type.
// Each bar = front face (rect) + top face (parallelogram).
// Bar height ∝ block_intensities[i]; zero-intensity bars are faint placeholders.
std::string FieldDag::svg_tensor_block(
    const FieldNode& n,
    const SVGPos& p,
    const FieldSvgProjectionNode* projected) const {
    std::ostringstream os;
    constexpr float BAR_W  = 7.0f;    // bar width
    constexpr float BAR_MIN = 3.0f;   // minimum visible bar height
    constexpr float BAR_MAX = 28.0f;  // max bar height
    constexpr float DEPTH  = 4.0f;    // isometric top-face depth
    constexpr float GAP    = 1.5f;
    const float total_w = 7 * (BAR_W + GAP) - GAP;
    const float start_x = p.x - total_w * 0.5f;

    for (uint8_t bi = 0; bi < 7; ++bi) {
        float h = BAR_MIN;
        float intensity = n.block_intensities[bi];
        if (projected && projected->bar_heights_valid) {
            h = std::max(BAR_MIN, projected->bar_heights[bi]);
            const float denom = std::max(0.001f, BAR_MAX - BAR_MIN);
            intensity = std::clamp((h - BAR_MIN) / denom, 0.0f, 1.0f);
        } else {
            h = std::max(BAR_MIN, n.block_intensities[bi] * BAR_MAX);
        }
        float bx = start_x + bi * (BAR_W + GAP);
        float by = p.y - 22.0f - h;   // base of bar (above node circle)
        const char* col = BLOCK_COLORS[bi];
        float alpha = intensity < 0.05f ? 0.18f : 0.90f;

        // Front face
        char buf[512];
        std::snprintf(buf, sizeof(buf),
            "<rect x=\"%.1f\" y=\"%.1f\" width=\"%.1f\" height=\"%.1f\""
            " fill=\"%s\" fill-opacity=\"%.2f\" rx=\"1\"/>",
            bx, by, BAR_W, h, col, alpha);
        os << buf;

        // Top face (isometric lid)
        if (intensity >= 0.05f) {
            // parallelogram: top-left, top-right, shifted-right, shifted-left
            std::snprintf(buf, sizeof(buf),
                "<polygon points=\"%.1f,%.1f %.1f,%.1f %.1f,%.1f %.1f,%.1f\""
                " fill=\"%s\" fill-opacity=\"0.55\"/>",
                bx,          by,
                bx + BAR_W,  by,
                bx + BAR_W + DEPTH * 0.7f, by - DEPTH * 0.5f,
                bx + DEPTH * 0.7f,         by - DEPTH * 0.5f,
                col);
            os << buf;
        }
    }
    return os.str();
}

// Render a tensor arc: shaded polygon connecting fold_map phase positions.
// Drawn behind track circles at Z-layer 0.5 (after field substrate, before edges).
std::string FieldDag::svg_tensor_arc(const FieldNode& n) const {
    if (n.fold_map.empty()) return {};
    std::ostringstream os;

    // Build point list from fold_map phases
    std::vector<SVGPos> pts;
    for (const auto& fe : n.fold_map) {
        kuhul::Phase ph = kuhul::Phase::Pop;
        kuhul::phase_from_name(fe.phase_name, ph);
        pts.push_back(phase_svg_pos(ph));
    }

    // Filled polygon (the tensor's multi-phase footprint)
    os << "<polygon points=\"";
    for (const auto& p : pts) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.1f,%.1f ", p.x, p.y);
        os << buf;
    }
    os << "\" fill=\"#38bdf8\" fill-opacity=\"0.08\""
       << " stroke=\"#38bdf8\" stroke-width=\"1.2\" stroke-opacity=\"0.55\""
       << " stroke-dasharray=\"4 3\"/>\n";

    // Small diamond at each fold_map position, colored by hook type
    for (size_t i = 0; i < n.fold_map.size(); ++i) {
        const auto& fe = n.fold_map[i];
        const SVGPos& p = pts[i];
        // hook color: on_read=primary, on_write=warning, monitor=danger, other=secondary
        const char* dc = "#38bdf8";
        if (fe.hook == "on_read")       dc = "#22f2b3";
        else if (fe.hook == "on_write") dc = "#facc15";
        else if (!fe.monitor.empty())   dc = "#fb7185";
        else if (fe.hook == "save_checkpoint" || fe.hook == "on_gc") dc = "#fbbf24";

        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "<polygon points=\"%.1f,%.1f %.1f,%.1f %.1f,%.1f %.1f,%.1f\""
            " fill=\"%s\" opacity=\"0.85\"/>\n",
            p.x,     p.y - 5,
            p.x + 5, p.y,
            p.x,     p.y + 5,
            p.x - 5, p.y,
            dc);
        os << buf;

        // Hook annotation label
        if (!fe.hook.empty() || !fe.monitor.empty()) {
            const char* lbl = fe.hook.empty() ? "monitor" : fe.hook.c_str();
            std::snprintf(buf, sizeof(buf),
                "<text x=\"%.1f\" y=\"%.1f\" text-anchor=\"middle\""
                " font-family=\"ui-monospace,monospace\" font-size=\"7\""
                " fill=\"%s\" opacity=\"0.75\">@%s</text>\n",
                p.x, p.y - 9.0f, dc, lbl);
            os << buf;
        }
    }

    // Centroid label (tensor id)
    float cx = 0, cy = 0;
    for (const auto& p : pts) { cx += p.x; cy += p.y; }
    cx /= (float)pts.size();
    cy /= (float)pts.size();
    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "<text x=\"%.1f\" y=\"%.1f\" text-anchor=\"middle\""
        " font-family=\"ui-monospace,monospace\" font-size=\"8\""
        " fill=\"#38bdf8\" opacity=\"0.7\">%s</text>\n",
        cx, cy + 4.0f,
        xml_escape(n.track_id).c_str());
    os << buf;

    return os.str();
}

std::string FieldDag::to_svg(const FieldSvgProjection* projection) const {
    // Build position map
    std::unordered_map<std::string, SVGPos> pos_map;
    std::unordered_map<std::string, const FieldSvgProjectionNode*> projected_nodes;
    const bool has_projection =
        projection && projection->nodes.size() == nodes_.size();

    for (size_t i = 0; i < nodes_.size(); ++i) {
        const auto& n = nodes_[i];
        if (has_projection) {
            const auto& pn = projection->nodes[i];
            pos_map[n.track_id] = {pn.x, pn.y};
            projected_nodes[n.track_id] = &pn;
        } else {
            pos_map[n.track_id] = to_svg_pos(n);
        }
    }

    std::ostringstream os;
    os << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
    os << "<svg xmlns=\"http://www.w3.org/2000/svg\""
       << " xmlns:k=\"https://khanary.io/tensor\""
       << " width=\"1024\" height=\"768\" viewBox=\"0 0 1024 768\">\n";
    os << "<title>KHANARY K-CUBE Semantic Field</title>\n";
    if (has_projection && projection->gpu_projected)
        os << "<desc>Field SVG projected via svg_compute.hlsl runtime path.</desc>\n";

    // ── Styles ────────────────────────────────────────────────────────────────
    os << R"(<style>
@keyframes pulse{0%,100%{opacity:.5}50%{opacity:1}}
@keyframes flow{0%{stroke-dashoffset:0}100%{stroke-dashoffset:-160}}
.fn{animation:pulse 2.5s ease-in-out infinite}
.ge{animation:flow 4s linear infinite}
</style>)";

    // ── Defs ──────────────────────────────────────────────────────────────────
    os << "<defs>"
       << "<radialGradient id=\"bg\" cx=\"50%\" cy=\"50%\" r=\"60%\">"
       << "<stop offset=\"0%\" stop-color=\"#050b1c\" stop-opacity=\"1\"/>"
       << "<stop offset=\"100%\" stop-color=\"#020617\" stop-opacity=\"1\"/>"
       << "</radialGradient>"
       << "<pattern id=\"grid\" width=\"40\" height=\"40\" patternUnits=\"userSpaceOnUse\">"
       << "<path d=\"M40 0L0 0 0 40\" fill=\"none\" stroke=\"#22f2b3\" stroke-width=\"0.25\" opacity=\"0.12\"/>"
       << "</pattern>"
       << "</defs>\n";

    // ── Layer 0: background + grid ────────────────────────────────────────────
    os << "<rect width=\"1024\" height=\"768\" fill=\"url(#bg)\"/>\n";
    os << "<rect width=\"1024\" height=\"768\" fill=\"url(#grid)\"/>\n";

    // ── Layer 0.5: tensor arcs (kind=tensor, multi-phase footprints) ─────────
    {
        bool any = false;
        for (const auto& n : nodes_) {
            if (n.kind == "tensor" && !n.fold_map.empty()) { any = true; break; }
        }
        if (any) {
            os << "<g id=\"tensor-arcs\">\n";
            for (const auto& n : nodes_)
                if (n.kind == "tensor") os << svg_tensor_arc(n);
            os << "</g>\n";
        }
    }

    // ── Layer 1: edges ────────────────────────────────────────────────────────
    os << "<g id=\"edges\" opacity=\"0.70\">\n";
    auto edge_pass = [](const FieldEdge& e) -> int {
        if (e.type == "vertical_fold")   return 0;
        if (e.type == "horizontal_fold") return 1;
        if (e.type == "ontology")        return 2;
        if (e.type == "delegate")        return 3;
        return 2;
    };
    // Draw from faint substrate edges to strong dispatch edges.
    for (int pass = 0; pass < 4; ++pass) {
        for (const auto& e : edges_) {
            if (edge_pass(e) != pass) continue;
            auto it_a = pos_map.find(e.from);
            auto it_b = pos_map.find(e.to);
            if (it_a == pos_map.end() || it_b == pos_map.end()) continue;
            const char* col = edge_color(e.type, e.verb);
            float op = 0.5f;
            float sw = 1.0f;
            if (e.type == "vertical_fold")      { op = 0.22f; sw = 1.0f; }
            else if (e.type == "horizontal_fold"){ op = 0.45f; sw = 1.0f; }
            else if (e.type == "ontology")      { op = 0.58f; sw = 1.2f; }
            else if (e.type == "delegate")      { op = 0.82f; sw = 1.5f; }

            bool is_relate = (micronaut::relation_matches(e.verb, "@related_to") || !e.directed);
            auto path = svg_edge_path(it_a->second, it_b->second);
            char buf[512];
            const char* dash = "5 4";
            if (is_relate) dash = "6 3 2 3";
            else if (e.type == "ontology") dash = "4 3";
            std::snprintf(buf, sizeof(buf),
                "  <path d=\"%s\" stroke=\"%s\" stroke-width=\"%.1f\""
                " fill=\"none\" stroke-dasharray=\"%s\""
                " opacity=\"%.2f\" class=\"ge\"",
                path.c_str(), col, sw, dash, op);
            os << buf
               << " k:edge_from=\""   << xml_escape(e.from)   << "\""
               << " k:edge_to=\""     << xml_escape(e.to)     << "\""
               << " k:verb=\""        << xml_escape(e.verb)   << "\""
               << " k:fold_type=\""   << xml_escape(e.type)   << "\""
               << " k:weight=\""      << e.weight             << "\""
               << " k:delta_theta=\"" << e.delta_theta        << "\""
               << "/>\n";
        }
    }
    os << "</g>\n";

    // ── Layer 2: SVG-3D tensor blocks ─────────────────────────────────────────
    os << "<g id=\"tensors\">\n";
    for (const auto& n : nodes_) {
        if (n.kind != "track") continue;
        auto it = pos_map.find(n.track_id);
        if (it == pos_map.end()) continue;
        const FieldSvgProjectionNode* pn = nullptr;
        auto pit = projected_nodes.find(n.track_id);
        if (pit != projected_nodes.end()) pn = pit->second;
        os << "  <g id=\"t_" << xml_escape(n.track_id) << "\">\n";
        os << "  " << svg_tensor_block(n, it->second, pn) << "\n";
        os << "  </g>\n";
    }
    os << "</g>\n";

    // ── Layer 3: field nodes ──────────────────────────────────────────────────
    os << "<g id=\"nodes\">\n";
    for (const auto& n : nodes_) {
        auto it = pos_map.find(n.track_id);
        if (it == pos_map.end()) continue;
        const SVGPos& p = it->second;

        kuhul::Phase ph = kuhul::Phase::Pop;
        kuhul::phase_from_name(n.phase_name, ph);
        const char* col = BLOCK_COLORS[static_cast<uint8_t>(phase_to_block(ph))];

        char buf[1024];
        if (n.kind == "ontology") {
            const float w = 152.0f;
            const float h = 44.0f;
            const float x = p.x - w * 0.5f;
            const float y = p.y - h * 0.5f;
            const char* fill = ontology_fill(n.track_id);
            std::snprintf(buf, sizeof(buf),
                "  <rect x=\"%.1f\" y=\"%.1f\" width=\"%.1f\" height=\"%.1f\" rx=\"5\""
                " fill=\"%s\" fill-opacity=\"0.70\" stroke=\"#e2e8f0\" stroke-width=\"0.8\"/>\n",
                x, y, w, h, fill);
            os << buf;
            std::snprintf(buf, sizeof(buf),
                "  <text x=\"%.1f\" y=\"%.1f\" text-anchor=\"middle\""
                " font-family=\"ui-monospace,monospace\" font-size=\"10\""
                " fill=\"#f8fafc\" font-weight=\"bold\">%s</text>\n",
                p.x, p.y - 4.0f, xml_escape(n.track_id).c_str());
            os << buf;
            std::string attrs;
            for (size_t i = 0; i < std::min<size_t>(3, n.effects.size()); ++i) {
                if (i) attrs += " | ";
                attrs += n.effects[i];
            }
            if (n.effects.size() > 3) attrs += " | ...";
            if (!attrs.empty()) {
                std::snprintf(buf, sizeof(buf),
                    "  <text x=\"%.1f\" y=\"%.1f\" text-anchor=\"middle\""
                    " font-family=\"ui-monospace,monospace\" font-size=\"7\""
                    " fill=\"#e2e8f0\" opacity=\"0.9\">%s</text>\n",
                    p.x, p.y + 10.0f, xml_escape(attrs).c_str());
                os << buf;
            }
            continue;
        }

        // Outer ring with k:* semantic attributes for SVG-Tensor round-trip
        std::snprintf(buf, sizeof(buf),
            "  <circle cx=\"%.1f\" cy=\"%.1f\" r=\"18\" fill=\"none\""
            " stroke=\"%s\" stroke-width=\"1.5\" opacity=\"0.7\" class=\"fn\"",
            p.x, p.y, col);
        os << buf
           << " k:track_id=\""   << xml_escape(n.track_id)   << "\""
           << " k:phase=\""      << xml_escape(n.phase_name) << "\""
           << " k:kind=\""       << xml_escape(n.kind)       << "\""
           << " k:entropy=\""    << n.local_entropy          << "\""
           << " k:geodesic_entropy=\"" << n.geodesic_entropy << "\""
           << " k:unruh_temp=\""  << n.unruh_temp            << "\""
           << " k:divergence=\""  << n.divergence            << "\""
           << " k:circumradius_R=\"" << n.circumradius_R     << "\""
           << " k:inradius_r=\""  << n.inradius_r            << "\""
           << " k:x=\""          << n.x                      << "\""
           << " k:y=\""          << n.y                      << "\""
           << " k:z=\""          << n.z                      << "\""
           << " k:weight=\""     << n.static_weight          << "\""
           << " k:confidence=\"" << n.confidence             << "\""
           << "/>\n";
        // Inner dot
        std::snprintf(buf, sizeof(buf),
            "  <circle cx=\"%.1f\" cy=\"%.1f\" r=\"5\" fill=\"%s\"/>\n",
            p.x, p.y, col);
        os << buf;
        // Coordinator marker
        if (n.coordinator) {
            std::snprintf(buf, sizeof(buf),
                "  <circle cx=\"%.1f\" cy=\"%.1f\" r=\"24\" fill=\"none\""
                " stroke=\"%s\" stroke-width=\"0.8\" stroke-dasharray=\"3 3\" opacity=\"0.4\"/>\n",
                p.x, p.y, col);
            os << buf;
        }
        // Label
        std::snprintf(buf, sizeof(buf),
            "  <text x=\"%.1f\" y=\"%.1f\" text-anchor=\"middle\""
            " font-family=\"ui-monospace,monospace\" font-size=\"9\""
            " fill=\"%s\">%s</text>\n",
            p.x, p.y + 30.0f, col, xml_escape(n.track_id).c_str());
        os << buf;
        std::snprintf(buf, sizeof(buf),
            "  <text x=\"%.1f\" y=\"%.1f\" text-anchor=\"middle\""
            " font-family=\"ui-monospace,monospace\" font-size=\"7\""
            " fill=\"%s\" opacity=\"0.6\">%s</text>\n",
            p.x, p.y + 40.0f, col, xml_escape(n.phase_name).c_str());
        os << buf;
    }
    os << "</g>\n";

    // ── HUD: legend ───────────────────────────────────────────────────────────
    os << "<g transform=\"translate(16,16)\">\n"
       << "  <rect width=\"180\" height=\"112\" rx=\"5\""
       << "  fill=\"rgba(2,6,23,0.88)\" stroke=\"#22f2b3\" stroke-width=\"0.8\"/>\n"
       << "  <text x=\"10\" y=\"16\" font-family=\"ui-monospace,monospace\""
       << "  font-size=\"9\" fill=\"#22f2b3\" font-weight=\"bold\">K-CUBE FIELD</text>\n";
    const char* bnames[7] = {"perceive","learn","think","evaluate","confidence","memory","evolve"};
    for (int i = 0; i < 7; ++i) {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "  <rect x=\"10\" y=\"%d\" width=\"8\" height=\"8\" fill=\"%s\"/>"
            "  <text x=\"24\" y=\"%d\" font-family=\"ui-monospace,monospace\""
            "  font-size=\"8\" fill=\"#e5f2ff\">%s</text>\n",
            24 + i * 12, BLOCK_COLORS[i], 32 + i * 12, bnames[i]);
        os << buf;
    }
    os << "</g>\n";

    os << "</svg>\n";
    return os.str();
}

// ── GPU geodesic propagator ────────────────────────────────────────────────────

namespace {

// C++ mirrors of the HLSL structs in geodesic_propagator.hlsl.
// Sizes must match exactly: FieldNode=52 bytes, FieldEdge=40 bytes.
#pragma pack(push, 4)
struct GpuFieldNode {
    float    x, y, z;           // float3 position
    float    confidence;
    float    static_weight;
    float    local_entropy;
    float    geodesic_entropy;
    float    unruh_temp;
    float    divergence;
    float    circumradius_R;
    float    inradius_r;
    uint32_t singularity_type;
    float    pad0;              // pad to 52 bytes
};
static_assert(sizeof(GpuFieldNode) == 52, "GpuFieldNode must be 52 bytes");

struct GpuFieldEdge {
    uint32_t source_id;
    uint32_t target_id;
    float    weight;
    float    length;
    float    vx, vy, vz;        // float3 velocity
    float    eh_x, eh_y, eh_z;  // float3 entropy_gradient
};
static_assert(sizeof(GpuFieldEdge) == 40, "GpuFieldEdge must be 40 bytes");

struct PropagatorConstants {
    float    kb_boltzmann;
    float    beta_diffusion_inv;
    uint32_t total_node_count;
    uint32_t total_edge_count;
};
#pragma pack(pop)

// Create a D3D11_RESOURCE_MISC_BUFFER_STRUCTURED buffer with an SRV.
static bool make_structured_srv(
    ID3D11Device* d,
    const void* data, uint32_t elem_count, uint32_t stride,
    ID3D11Buffer** out_buf,
    ID3D11ShaderResourceView** out_srv)
{
    D3D11_BUFFER_DESC bd{};
    bd.ByteWidth           = stride * elem_count;
    bd.Usage               = D3D11_USAGE_DEFAULT;
    bd.BindFlags           = D3D11_BIND_SHADER_RESOURCE;
    bd.MiscFlags           = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    bd.StructureByteStride = stride;
    D3D11_SUBRESOURCE_DATA init{data, 0, 0};
    if (FAILED(d->CreateBuffer(&bd, data ? &init : nullptr, out_buf))) return false;

    D3D11_SHADER_RESOURCE_VIEW_DESC sd{};
    sd.Format                   = DXGI_FORMAT_UNKNOWN;
    sd.ViewDimension            = D3D11_SRV_DIMENSION_BUFFEREX;
    sd.BufferEx.NumElements     = elem_count;
    return SUCCEEDED(d->CreateShaderResourceView(*out_buf, &sd, out_srv));
}

// Create a D3D11_RESOURCE_MISC_BUFFER_STRUCTURED RW buffer with a UAV.
static bool make_structured_uav(
    ID3D11Device* d,
    uint32_t elem_count, uint32_t stride,
    ID3D11Buffer** out_buf,
    ID3D11UnorderedAccessView** out_uav)
{
    D3D11_BUFFER_DESC bd{};
    bd.ByteWidth           = stride * elem_count;
    bd.Usage               = D3D11_USAGE_DEFAULT;
    bd.BindFlags           = D3D11_BIND_UNORDERED_ACCESS;
    bd.MiscFlags           = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    bd.StructureByteStride = stride;
    if (FAILED(d->CreateBuffer(&bd, nullptr, out_buf))) return false;

    D3D11_UNORDERED_ACCESS_VIEW_DESC ud{};
    ud.Format              = DXGI_FORMAT_UNKNOWN;
    ud.ViewDimension       = D3D11_UAV_DIMENSION_BUFFER;
    ud.Buffer.NumElements  = elem_count;
    return SUCCEEDED(d->CreateUnorderedAccessView(*out_buf, &ud, out_uav));
}

static bool ensure_dynamic_hyperbolic_attention_srv(
    ID3D11Device* d,
    uint32_t required_records,
    uint32_t extra_capacity_records,
    HyperbolicAttentionGpuBuffer& out)
{
    if (!d || required_records == 0) return false;

    if (out.structured_buffer && out.srv &&
        out.capacity_records >= required_records &&
        out.stride_bytes == sizeof(HyperbolicAttentionRecord64)) {
        return true;
    }

    const uint32_t headroom =
        (std::numeric_limits<uint32_t>::max() - required_records > extra_capacity_records)
            ? extra_capacity_records
            : 0u;
    const uint32_t capacity = required_records + headroom;
    if (capacity == 0 ||
        capacity > std::numeric_limits<uint32_t>::max() / sizeof(HyperbolicAttentionRecord64)) {
        return false;
    }

    Microsoft::WRL::ComPtr<ID3D11Buffer> new_buffer;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> new_srv;

    D3D11_BUFFER_DESC bd{};
    bd.ByteWidth           = capacity * static_cast<UINT>(sizeof(HyperbolicAttentionRecord64));
    bd.Usage               = D3D11_USAGE_DYNAMIC;
    bd.BindFlags           = D3D11_BIND_SHADER_RESOURCE;
    bd.CPUAccessFlags      = D3D11_CPU_ACCESS_WRITE;
    bd.MiscFlags           = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    bd.StructureByteStride = static_cast<UINT>(sizeof(HyperbolicAttentionRecord64));
    if (FAILED(d->CreateBuffer(&bd, nullptr, &new_buffer))) return false;

    D3D11_SHADER_RESOURCE_VIEW_DESC sd{};
    sd.Format               = DXGI_FORMAT_UNKNOWN;
    sd.ViewDimension        = D3D11_SRV_DIMENSION_BUFFEREX;
    sd.BufferEx.FirstElement = 0;
    sd.BufferEx.NumElements = capacity;
    if (FAILED(d->CreateShaderResourceView(new_buffer.Get(), &sd, &new_srv)))
        return false;

    out.structured_buffer = new_buffer;
    out.srv = new_srv;
    out.capacity_records = capacity;
    out.stride_bytes = static_cast<uint32_t>(sizeof(HyperbolicAttentionRecord64));
    return true;
}

static float map_lookup_or(const std::unordered_map<std::string, float>& map,
                           const std::string& key,
                           float fallback) {
    const auto it = map.find(key);
    return it == map.end() ? fallback : it->second;
}

} // anonymous namespace

bool FieldDag::upload_hyperbolic_attention_records(
    gpu::D3D11Device& dev,
    const std::vector<HyperbolicAttentionRecord64>& records,
    HyperbolicAttentionGpuBuffer& out,
    uint32_t extra_capacity_records)
{
    out.record_count = 0;
    if (!dev.is_valid() || records.empty()) return false;

    auto* d = dev.GetDevice();
    auto* ctx = dev.GetContext();
    if (records.size() > std::numeric_limits<uint32_t>::max()) return false;
    const uint32_t count = static_cast<uint32_t>(records.size());

    if (!ensure_dynamic_hyperbolic_attention_srv(d, count, extra_capacity_records, out))
        return false;

    D3D11_MAPPED_SUBRESOURCE mapped{};
    if (FAILED(ctx->Map(out.structured_buffer.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped)))
        return false;

    const size_t copy_bytes = records.size() * sizeof(HyperbolicAttentionRecord64);
    std::memcpy(mapped.pData, records.data(), copy_bytes);

    const size_t capacity_bytes =
        static_cast<size_t>(out.capacity_records) * sizeof(HyperbolicAttentionRecord64);
    if (capacity_bytes > copy_bytes) {
        auto* spare = static_cast<unsigned char*>(mapped.pData) + copy_bytes;
        std::memset(spare, 0, capacity_bytes - copy_bytes);
    }

    ctx->Unmap(out.structured_buffer.Get(), 0);
    out.record_count = count;
    return true;
}

bool FieldDag::upload_hyperbolic_trajectories(
    gpu::D3D11Device& dev,
    const std::vector<HyperbolicEventVector>& events,
    const std::unordered_map<std::string, float>& wcr_w_map,
    const std::unordered_map<std::string, float>& wcr_c_map,
    const std::unordered_map<std::string, float>& wcr_r_map,
    HyperbolicAttentionGpuBuffer& out,
    uint32_t extra_capacity_records)
{
    out.record_count = 0;
    if (!dev.is_valid() || events.empty()) return false;

    auto* d = dev.GetDevice();
    auto* ctx = dev.GetContext();
    if (events.size() > std::numeric_limits<uint32_t>::max()) return false;
    const uint32_t count = static_cast<uint32_t>(events.size());

    if (!ensure_dynamic_hyperbolic_attention_srv(d, count, extra_capacity_records, out))
        return false;

    D3D11_MAPPED_SUBRESOURCE mapped{};
    if (FAILED(ctx->Map(out.structured_buffer.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped)))
        return false;

    auto* dest = static_cast<HyperbolicAttentionRecord64*>(mapped.pData);
    for (uint32_t i = 0; i < count; ++i) {
        const auto& event = events[i];
        dest[i] = make_hyperbolic_attention_record(
            event,
            map_lookup_or(wcr_w_map, event.key, 1.0f),
            map_lookup_or(wcr_c_map, event.key, 1.0f),
            map_lookup_or(wcr_r_map, event.key, 0.5f));
    }

    if (out.capacity_records > count) {
        std::memset(dest + count, 0,
                    static_cast<size_t>(out.capacity_records - count) *
                    sizeof(HyperbolicAttentionRecord64));
    }

    ctx->Unmap(out.structured_buffer.Get(), 0);
    out.record_count = count;
    return true;
}

PropagatorDispatchResult FieldDag::dispatch_geodesic_propagator(
    gpu::D3D11Device& dev, ID3DBlob* cso, float beta_inv, float kb_boltzmann)
{
    PropagatorDispatchResult res;
    if (!dev.is_valid() || !cso || nodes_.empty()) return res;

    auto* d   = dev.GetDevice();
    auto* ctx = dev.GetContext();
    const uint32_t N = static_cast<uint32_t>(nodes_.size());

    // ── Build node index ──────────────────────────────────────────────────────
    std::unordered_map<std::string, uint32_t> idx;
    idx.reserve(N);
    for (uint32_t i = 0; i < N; ++i) idx[nodes_[i].track_id] = i;

    // ── Pack GpuFieldNode ─────────────────────────────────────────────────────
    std::vector<GpuFieldNode> gnodes(N);
    for (uint32_t i = 0; i < N; ++i) {
        const auto& n   = nodes_[i];
        GpuFieldNode& g = gnodes[i];
        g.x = n.x; g.y = n.y; g.z = n.z;
        g.confidence      = n.confidence;
        g.static_weight   = n.static_weight;
        g.local_entropy   = n.local_entropy;
        g.geodesic_entropy = n.geodesic_entropy;
        g.unruh_temp      = n.unruh_temp;
        g.divergence      = n.divergence;
        g.circumradius_R  = n.circumradius_R;
        g.inradius_r      = n.inradius_r;
        g.singularity_type = n.singularity_type;
        g.pad0 = 0.0f;
    }

    // ── Pack GpuFieldEdge (skip edges with unknown node ids) ──────────────────
    std::vector<GpuFieldEdge> gedges;
    gedges.reserve(edges_.size());
    for (const auto& e : edges_) {
        auto si = idx.find(e.from);
        auto ti = idx.find(e.to);
        if (si == idx.end() || ti == idx.end()) continue;
        GpuFieldEdge ge;
        ge.source_id = si->second;
        ge.target_id = ti->second;
        ge.weight    = e.weight;
        ge.length    = e.length;
        ge.vx = e.vx; ge.vy = e.vy; ge.vz = e.vz;
        ge.eh_x = e.eh_x; ge.eh_y = e.eh_y; ge.eh_z = e.eh_z;
        gedges.push_back(ge);
    }
    const uint32_t E = static_cast<uint32_t>(gedges.size());

    // ── Create shader ─────────────────────────────────────────────────────────
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> cs;
    if (FAILED(d->CreateComputeShader(
            cso->GetBufferPointer(), cso->GetBufferSize(), nullptr, &cs)))
        return res;

    // ── NodeIn (t0): read-only structured buffer ──────────────────────────────
    Microsoft::WRL::ComPtr<ID3D11Buffer>             node_in_buf;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> node_in_srv;
    if (!make_structured_srv(d, gnodes.data(), N, sizeof(GpuFieldNode),
                             &node_in_buf, &node_in_srv))
        return res;

    // ── EdgeIn (t1): read-only structured buffer ──────────────────────────────
    Microsoft::WRL::ComPtr<ID3D11Buffer>             edge_in_buf;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> edge_in_srv;
    if (E > 0) {
        if (!make_structured_srv(d, gedges.data(), E, sizeof(GpuFieldEdge),
                                 &edge_in_buf, &edge_in_srv))
            return res;
    }

    // ── NodeOut (u0): RW structured buffer — pre-loaded with NodeIn data ──────
    Microsoft::WRL::ComPtr<ID3D11Buffer>              node_out_buf;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> node_out_uav;
    if (!make_structured_uav(d, N, sizeof(GpuFieldNode),
                             &node_out_buf, &node_out_uav))
        return res;
    ctx->CopyResource(node_out_buf.Get(), node_in_buf.Get());

    // ── PropagatorConstants cbuffer (b0) ──────────────────────────────────────
    PropagatorConstants pc;
    pc.kb_boltzmann       = kb_boltzmann;
    pc.beta_diffusion_inv = beta_inv;
    pc.total_node_count   = N;
    pc.total_edge_count   = E;

    Microsoft::WRL::ComPtr<ID3D11Buffer> cbuf;
    {
        D3D11_BUFFER_DESC bd{};
        bd.ByteWidth      = (sizeof(PropagatorConstants) + 15u) & ~15u; // 16-byte align
        bd.Usage          = D3D11_USAGE_DYNAMIC;
        bd.BindFlags      = D3D11_BIND_CONSTANT_BUFFER;
        bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        D3D11_SUBRESOURCE_DATA init{&pc, 0, 0};
        if (FAILED(d->CreateBuffer(&bd, &init, &cbuf))) return res;
    }

    // ── Dispatch ──────────────────────────────────────────────────────────────
    ctx->CSSetShader(cs.Get(), nullptr, 0);
    ctx->CSSetConstantBuffers(0, 1, cbuf.GetAddressOf());

    ID3D11ShaderResourceView* srvs[] = {node_in_srv.Get(),
                                        E > 0 ? edge_in_srv.Get() : nullptr};
    ctx->CSSetShaderResources(0, 2, srvs);

    const uint32_t INITIAL_COUNT = 0xFFFFFFFFu;
    ctx->CSSetUnorderedAccessViews(0, 1, node_out_uav.GetAddressOf(), &INITIAL_COUNT);

    ctx->Dispatch((N + 63u) / 64u, 1u, 1u);

    // ── Unbind resources ──────────────────────────────────────────────────────
    ID3D11ShaderResourceView* null_srvs[2] = {nullptr, nullptr};
    ctx->CSSetShaderResources(0, 2, null_srvs);
    ID3D11UnorderedAccessView* null_uav = nullptr;
    ctx->CSSetUnorderedAccessViews(0, 1, &null_uav, nullptr);
    ctx->CSSetShader(nullptr, nullptr, 0);

    // ── Readback via staging buffer ───────────────────────────────────────────
    Microsoft::WRL::ComPtr<ID3D11Buffer> staging;
    {
        D3D11_BUFFER_DESC bd{};
        bd.ByteWidth           = sizeof(GpuFieldNode) * N;
        bd.Usage               = D3D11_USAGE_STAGING;
        bd.CPUAccessFlags      = D3D11_CPU_ACCESS_READ;
        bd.MiscFlags           = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        bd.StructureByteStride = sizeof(GpuFieldNode);
        if (FAILED(d->CreateBuffer(&bd, nullptr, &staging))) return res;
    }
    ctx->CopyResource(staging.Get(), node_out_buf.Get());

    D3D11_MAPPED_SUBRESOURCE mapped{};
    if (FAILED(ctx->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped))) return res;

    const auto* out_nodes = static_cast<const GpuFieldNode*>(mapped.pData);
    res.geodesic_entropy.resize(N);
    for (uint32_t i = 0; i < N; ++i)
        res.geodesic_entropy[i] = out_nodes[i].geodesic_entropy;
    ctx->Unmap(staging.Get(), 0);

    // ── Write updated geodesic_entropy back onto FieldNodes ───────────────────
    for (uint32_t i = 0; i < N; ++i)
        nodes_[i].geodesic_entropy = res.geodesic_entropy[i];

    // ── Promote substrate descriptors to GPU_VRAM_D3D11 ───────────────────────
    // The D3D11 buffers are temporaries this dispatch; physical_buffer_allocation
    // stays null.  Residency flag records that the mesh has been GPU-processed.
    for (auto& [pid, ts] : substrates_) {
        ts.residency = DeviceResidency::GPU_VRAM_D3D11;
        ts.shape     = { static_cast<uint32_t>(N), sizeof(GpuFieldNode) / 4 };
    }

    res.gpu_ran = true;
    return res;
}

// ── Ch'en Raychaudhuri gate ───────────────────────────────────────────────────
// Integrates a discrete approximation of dH_G/dt = Σ(R_μν u^μ u^ν + ⅓θ²) along
// the path.  Approximation: rate_e ≈ H_G_avg * (1 + div_avg/3) per step, where
// div < 0 (sink focusing) suppresses the rate and div > 0 (source expansion) amplifies it.
// Ch'en drops the branch when the integral exceeds hg_max.

bool FieldDag::check_raychaudhuri_gate(
    const std::vector<std::string>& path, float hg_max) const
{
    if (path.empty()) return true;

    std::unordered_map<std::string, size_t> idx;
    idx.reserve(nodes_.size());
    for (size_t i = 0; i < nodes_.size(); ++i)
        idx[nodes_[i].track_id] = i;

    float integral = 0.0f;
    for (size_t k = 0; k + 1 < path.size(); ++k) {
        auto ai = idx.find(path[k]);
        auto bi = idx.find(path[k + 1]);
        if (ai == idx.end() || bi == idx.end()) continue;
        const auto& na = nodes_[ai->second];
        const auto& nb = nodes_[bi->second];
        // Average H_G and divergence (expansion scalar θ ≈ div V) over the step.
        const float hg_avg  = 0.5f * (na.geodesic_entropy + nb.geodesic_entropy);
        const float div_avg = 0.5f * (na.divergence        + nb.divergence);
        // dH_G/dt ≈ H_G * (1 + θ/3); Δt = 1 per discrete step.
        integral += hg_avg * (1.0f + div_avg / 3.0f);
    }
    return integral <= hg_max;
}

// ── DirectionStore → cbuffer feed ─────────────────────────────────────────────
// Translates a co-occurrence delta ΔM into an updated β⁻¹ for PropagatorConstants.
// Higher |ΔM| → more diffusion → higher β⁻¹ (more exploration).
// tanh saturates the adjustment so β⁻¹ cannot grow unboundedly.
// The decay factor reduces sensitivity as the training step count grows,
// stabilising the geometry when the provider has already converged.

float FieldDag::compute_beta_inv_from_delta_m(
    float delta_m, float base_beta_inv, uint32_t step)
{
    const float decay = 1.0f / (1.0f + static_cast<float>(step) * 0.01f);
    return base_beta_inv * (1.0f + std::tanh(delta_m * decay));
}

} // namespace khanary::world
