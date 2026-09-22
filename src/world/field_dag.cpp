#include "field_dag.h"
#include "../kuhul/phase_engine.h"
#include <sstream>
#include <cmath>
#include <unordered_map>
#include <unordered_set>

namespace khanary::world {

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
        fn.confidence  = e->confidence;
        fn.x           = wn->x;
        fn.y           = wn->y;
        fn.z           = wn->z;
        fn.kind              = "track";
        fn.coordinator       = e->sandbox.coordinator;
        fn.effects           = e->sandbox.effects;
        fn.block_intensities = wn->block_intensities;

        auto fi = static_cast<size_t>(e->primary_phase);
        fn.face_name = (fi < KCUBE_FACES.size())
                           ? std::string(KCUBE_FACES[fi].geometry)
                           : fn.phase_name;

        phase_to_tracks[static_cast<int>(e->primary_phase)].push_back(id);
        nodes_.push_back(std::move(fn));
    }

    // ── Edge set 1: declared delegate arcs ──────────────────────────────────
    for (const auto& id : reg_.ids()) {
        const auto* e = reg_.get(id);
        if (!e || e->sub_micronauts.empty()) continue;
        for (const auto& target : e->sub_micronauts) {
            if (node_map.count(target)) {
                FieldEdge edge;
                edge.from          = id;
                edge.to            = target;
                edge.type          = "delegate";
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
    for (const auto& n : nodes_) {
        os << "    <node"
           << " id=\""          << xml_escape(n.track_id)   << "\""
           << " kind=\""        << xml_escape(n.kind)        << "\""
           << " phase=\""       << xml_escape(n.phase_name) << "\""
           << " face=\""        << xml_escape(n.face_name)  << "\""
           << " block=\""       << xml_escape(n.block_name) << "\""
           << " confidence=\""  << n.confidence             << "\""
           << " coordinator=\"" << (n.coordinator ? "true" : "false") << "\""
           << " x=\""           << n.x << "\""
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
           << " from=\""          << xml_escape(e.from)  << "\""
           << " to=\""            << xml_escape(e.to)    << "\""
           << " type=\""          << xml_escape(e.type)  << "\""
           << " weight=\""        << e.weight             << "\""
           << " delta_theta=\""   << static_cast<int>(e.delta_theta) << "\""
           << " xcfe_required=\"" << (e.xcfe_required ? "true" : "false") << "\""
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

// Edge type colors
static const char* edge_color(const std::string& type) {
    if (type == "delegate")       return "#22f2b3";
    if (type == "horizontal_fold") return "#38bdf8";
    return "#6b7280";  // vertical_fold — faint
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
std::string FieldDag::svg_tensor_block(const FieldNode& n, const SVGPos& p) const {
    std::ostringstream os;
    constexpr float BAR_W  = 7.0f;    // bar width
    constexpr float BAR_MAX = 28.0f;  // max bar height
    constexpr float DEPTH  = 4.0f;    // isometric top-face depth
    constexpr float GAP    = 1.5f;
    const float total_w = 7 * (BAR_W + GAP) - GAP;
    const float start_x = p.x - total_w * 0.5f;

    for (uint8_t bi = 0; bi < 7; ++bi) {
        float h = std::max(3.0f, n.block_intensities[bi] * BAR_MAX);
        float bx = start_x + bi * (BAR_W + GAP);
        float by = p.y - 22.0f - h;   // base of bar (above node circle)
        const char* col = BLOCK_COLORS[bi];
        float alpha = n.block_intensities[bi] < 0.05f ? 0.18f : 0.90f;

        // Front face
        char buf[512];
        std::snprintf(buf, sizeof(buf),
            "<rect x=\"%.1f\" y=\"%.1f\" width=\"%.1f\" height=\"%.1f\""
            " fill=\"%s\" fill-opacity=\"%.2f\" rx=\"1\"/>",
            bx, by, BAR_W, h, col, alpha);
        os << buf;

        // Top face (isometric lid)
        if (n.block_intensities[bi] >= 0.05f) {
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

std::string FieldDag::to_svg() const {
    // Build position map
    std::unordered_map<std::string, SVGPos> pos_map;
    for (const auto& n : nodes_)
        pos_map[n.track_id] = to_svg_pos(n);

    std::ostringstream os;
    os << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
    os << "<svg xmlns=\"http://www.w3.org/2000/svg\""
       << " width=\"1024\" height=\"768\" viewBox=\"0 0 1024 768\">\n";
    os << "<title>KHANARY K-CUBE Semantic Field</title>\n";

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
    os << "<g id=\"edges\" opacity=\"0.65\">\n";
    // Draw vertical_fold first (faintest), then horizontal, then delegate on top
    for (int pass = 0; pass < 3; ++pass) {
        const char* type_filter[] = {"vertical_fold", "horizontal_fold", "delegate"};
        for (const auto& e : edges_) {
            if (e.type != type_filter[pass]) continue;
            auto it_a = pos_map.find(e.from);
            auto it_b = pos_map.find(e.to);
            if (it_a == pos_map.end() || it_b == pos_map.end()) continue;
            const char* col = edge_color(e.type);
            float op   = (e.type == "vertical_fold")    ? 0.22f
                       : (e.type == "horizontal_fold")  ? 0.45f
                       :                                   0.80f;
            float sw   = (e.type == "delegate") ? 1.5f : 1.0f;
            auto path  = svg_edge_path(it_a->second, it_b->second);
            char buf[512];
            std::snprintf(buf, sizeof(buf),
                "  <path d=\"%s\" stroke=\"%s\" stroke-width=\"%.1f\""
                " fill=\"none\" stroke-dasharray=\"5 4\""
                " opacity=\"%.2f\" class=\"ge\"/>\n",
                path.c_str(), col, sw, op);
            os << buf;
        }
    }
    os << "</g>\n";

    // ── Layer 2: SVG-3D tensor blocks ─────────────────────────────────────────
    os << "<g id=\"tensors\">\n";
    for (const auto& n : nodes_) {
        auto it = pos_map.find(n.track_id);
        if (it == pos_map.end()) continue;
        os << "  <g id=\"t_" << xml_escape(n.track_id) << "\">\n";
        os << "  " << svg_tensor_block(n, it->second) << "\n";
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
        // Outer ring
        std::snprintf(buf, sizeof(buf),
            "  <circle cx=\"%.1f\" cy=\"%.1f\" r=\"18\" fill=\"none\""
            " stroke=\"%s\" stroke-width=\"1.5\" opacity=\"0.7\" class=\"fn\"/>\n",
            p.x, p.y, col);
        os << buf;
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

} // namespace khanary::world
