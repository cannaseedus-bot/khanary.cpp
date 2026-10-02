#include "stb_map_convert.h"
#include <unordered_map>

namespace khanary::io {

static uint32_t fold_type_id(const std::string& s) {
    if (s == "vertical"   || s == "vertical_fold")   return 0;
    if (s == "horizontal" || s == "horizontal_fold") return 1;
    if (s == "delegate")                              return 2;
    if (s == "ontology")                              return 3;
    return 4;
}

static void pad3(float (&dst)[3], const std::vector<float>& src) {
    dst[0] = src.size() > 0 ? src[0] : 0.0f;
    dst[1] = src.size() > 1 ? src[1] : 0.0f;
    dst[2] = src.size() > 2 ? src[2] : 0.0f;
}

gpu::STBMap svg_tensor_to_stb_map(const SVGTensorGraph& g) {
    std::unordered_map<std::string, uint32_t> id_map;
    for (size_t i = 0; i < g.nodes.size(); ++i)
        id_map[g.nodes[i].track_id] = static_cast<uint32_t>(i);

    std::vector<gpu::STBFieldRecord> field_recs;
    for (const auto& n : g.nodes) {
        if (n.q_vec.empty() && n.k_vec.empty() && n.v_vec.empty()) continue;
        gpu::STBFieldRecord r{};
        pad3(r.q, n.q_vec);
        pad3(r.k, n.k_vec);
        pad3(r.v, n.v_vec);
        field_recs.push_back(r);
    }

    std::vector<gpu::STBGeodesicRecord> geo_recs;
    for (const auto& n : g.nodes) {
        gpu::STBGeodesicRecord r{};
        r.x = n.x; r.y = n.y; r.z = n.z;
        r.local_entropy    = n.entropy;
        r.geodesic_entropy = n.geodesic_entropy;
        r.unruh_temp       = n.unruh_temp;
        r.divergence       = n.divergence;
        r.circumradius_R   = n.circumradius_R;
        r.inradius_r       = n.inradius_r;
        geo_recs.push_back(r);
    }

    std::vector<gpu::STBFoldRecord> fold_recs;
    for (const auto& e : g.edges) {
        gpu::STBFoldRecord r{};
        const auto it_a = id_map.find(e.from);
        const auto it_b = id_map.find(e.to);
        r.from_phase  = (it_a != id_map.end()) ? it_a->second : ~uint32_t{0};
        r.to_phase    = (it_b != id_map.end()) ? it_b->second : ~uint32_t{0};
        r.fold_type   = fold_type_id(e.fold_type);
        r.arc_weight  = e.weight;
        r.delta_theta = e.delta_theta;
        fold_recs.push_back(r);
    }

    gpu::STBMap m;
    if (!field_recs.empty()) m.sections.push_back(gpu::make_field_section(field_recs));
    if (!geo_recs.empty())   m.sections.push_back(gpu::make_geodesic_section(geo_recs));
    if (!fold_recs.empty())  m.sections.push_back(gpu::make_fold_section(fold_recs));
    return m;
}

gpu::STBMap field_vectors_to_stb_map(
    const std::vector<world::FieldNode>& nodes,
    const std::vector<world::FieldEdge>& edges)
{
    std::unordered_map<std::string, uint32_t> id_map;
    for (size_t i = 0; i < nodes.size(); ++i)
        id_map[nodes[i].track_id] = static_cast<uint32_t>(i);

    std::vector<gpu::STBGeodesicRecord> geo_recs;
    for (const auto& n : nodes) {
        gpu::STBGeodesicRecord r{};
        r.x = n.x; r.y = n.y; r.z = n.z;
        r.local_entropy    = n.local_entropy;
        r.geodesic_entropy = n.geodesic_entropy;
        r.unruh_temp       = n.unruh_temp;
        r.divergence       = n.divergence;
        r.circumradius_R   = n.circumradius_R;
        r.inradius_r       = n.inradius_r;
        geo_recs.push_back(r);
    }

    std::vector<gpu::STBFoldRecord> fold_recs;
    for (const auto& e : edges) {
        gpu::STBFoldRecord r{};
        const auto it_a = id_map.find(e.from);
        const auto it_b = id_map.find(e.to);
        r.from_phase  = (it_a != id_map.end()) ? it_a->second : ~uint32_t{0};
        r.to_phase    = (it_b != id_map.end()) ? it_b->second : ~uint32_t{0};
        r.fold_type   = fold_type_id(e.type);
        r.arc_weight  = e.weight;
        r.delta_theta = e.delta_theta;
        fold_recs.push_back(r);
    }

    gpu::STBMap m;
    if (!geo_recs.empty())  m.sections.push_back(gpu::make_geodesic_section(geo_recs));
    if (!fold_recs.empty()) m.sections.push_back(gpu::make_fold_section(fold_recs));
    return m;
}

} // namespace khanary::io
