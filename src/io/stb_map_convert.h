#pragma once
#include "svg_tensor.h"
#include "../gpu/stb_map.h"
#include "../world/field_dag.h"

namespace khanary::io {

// Convert an in-memory SVGTensorGraph → STBMap.
//   Field section    : one record per node with non-empty Q/K/V vectors
//   Geodesic section : one record per node (x/y/z + entropy fields)
//   Fold section     : one record per edge
// Sections are omitted if the corresponding input collection is empty.
gpu::STBMap svg_tensor_to_stb_map(const SVGTensorGraph& g);

// Convert raw FieldNode/FieldEdge vectors → STBMap.
//   Geodesic section : one record per node (geometry + entropy)
//   Fold section     : one record per edge
// Does not produce a Field section (FieldNode has no Q/K/V vectors).
gpu::STBMap field_vectors_to_stb_map(
    const std::vector<world::FieldNode>& nodes,
    const std::vector<world::FieldEdge>& edges);

// Thin wrapper: FieldDag → STBMap via field_vectors_to_stb_map.
inline gpu::STBMap field_dag_to_stb_map(const world::FieldDag& dag) {
    return field_vectors_to_stb_map(dag.nodes(), dag.edges());
}

} // namespace khanary::io
