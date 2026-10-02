#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cctype>

namespace khanary::kuhul {

// P_M(ΔF): co-occurrence mass changes — ΔM_{Y,X}
// For factory evolution episodes this is always empty. M_{Y,X} updates come
// from full query→response cycle extractors that observe both fold fields.
struct CooccurrenceDelta {
    struct Entry { std::string Y; std::string X; float delta = 1.0f; };
    std::vector<Entry> entries;
    bool empty() const { return entries.empty(); }
};

// P_R(ΔF): typed relation edge changes — ΔRelations
// Populated by factory evolution episodes (capability and governs extensions).
struct RelationDelta {
    struct CapEdge { std::string micronaut_id; std::string cap_token; };
    struct GovEdge { std::string micronaut_id; std::string scope_token; };
    std::vector<CapEdge> added_capabilities;
    std::vector<GovEdge> added_governs;
    bool empty() const { return added_capabilities.empty() && added_governs.empty(); }
};

// P_W(ΔF): static_weight change — ΔW
struct WeightDelta {
    std::string micronaut_id;
    float       delta_W = 0.0f;
    bool empty() const { return delta_W == 0.0f; }
};

// P_C(ΔF): evidence-based confidence change — ΔC
struct ConfidenceDelta {
    std::string micronaut_id;
    float       delta_C = 0.0f;
    bool empty() const { return delta_C == 0.0f; }
};

// VECTOR-µ relation-conditioned geometry support.
//
// These structs describe derived representation learning over ΔF evidence.
// They do not authorize semantic mutation. RelationISA/FoldGraph remain the
// source of truth; VECTOR-µ only learns useful coordinates over that structure.
enum class RelationGeometryKind {
    Similarity,     // symmetric neighborhood score, e.g. @related_to
    Translational,  // directional shift, e.g. @causes, @requires
    Bilinear,       // typed subspace projection, e.g. @is_a, @part_of
    Rotational,     // contrast/opposition plane, e.g. @contradicts
    Structural,     // control/context graph relation, e.g. @governs, @via
    Unknown
};

inline const char* relation_geometry_kind_name(RelationGeometryKind kind) {
    switch (kind) {
    case RelationGeometryKind::Similarity:    return "similarity";
    case RelationGeometryKind::Translational: return "translational";
    case RelationGeometryKind::Bilinear:      return "bilinear";
    case RelationGeometryKind::Rotational:    return "rotational";
    case RelationGeometryKind::Structural:    return "structural";
    default:                                  return "unknown";
    }
}

inline std::string normalize_relation_geometry_token(std::string relation) {
    std::transform(relation.begin(), relation.end(), relation.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    std::replace(relation.begin(), relation.end(), '-', '_');
    if (!relation.empty() && relation.front() != '@')
        relation.insert(relation.begin(), '@');
    return relation;
}

inline RelationGeometryKind relation_geometry_kind_for(std::string relation) {
    relation = normalize_relation_geometry_token(std::move(relation));
    if (relation == "@related_to" || relation == "@supports" ||
        relation == "@evidence_for")
        return RelationGeometryKind::Similarity;
    if (relation == "@causes" || relation == "@prevents" ||
        relation == "@enables" || relation == "@requires" ||
        relation == "@depends_on" || relation == "@produces" ||
        relation == "@delegates_to" || relation == "@calls")
        return RelationGeometryKind::Translational;
    if (relation == "@is_a" || relation == "@part_of" ||
        relation == "@has_a" || relation == "@has_part" ||
        relation == "@has_property" || relation == "@contains")
        return RelationGeometryKind::Bilinear;
    if (relation == "@contradicts")
        return RelationGeometryKind::Rotational;
    if (relation == "@governs" || relation == "@via" ||
        relation == "@context" || relation == "@previous" ||
        relation == "@next" || relation == "@return_to" ||
        relation == "@resumes")
        return RelationGeometryKind::Structural;
    return RelationGeometryKind::Unknown;
}

inline bool relation_geometry_is_asymmetric(std::string relation) {
    const auto kind = relation_geometry_kind_for(relation);
    return kind == RelationGeometryKind::Translational ||
           kind == RelationGeometryKind::Bilinear ||
           kind == RelationGeometryKind::Structural;
}

struct RelationConditionedGeometrySpec {
    std::string relation = "@related_to";
    RelationGeometryKind kind = RelationGeometryKind::Similarity;
    bool asymmetric = false;

    // Projection-only invariant: the learned geometry may rank/retrieve, but it
    // cannot redefine the typed relation supplied by RelationISA/FoldGraph.
    bool semantic_authority = false;
    bool projection_only = true;

    // Contract notation for VECTOR-µ providers:
    // R_r(A,B) = f(v_A, v_B | relation, provider, context).
    std::string scoring_formula = "R_r(A,B)=f(v_A,v_B|relation,provider,context)";

    static RelationConditionedGeometrySpec from_relation(std::string rel) {
        RelationConditionedGeometrySpec spec;
        spec.relation = normalize_relation_geometry_token(std::move(rel));
        spec.kind = relation_geometry_kind_for(spec.relation);
        spec.asymmetric = relation_geometry_is_asymmetric(spec.relation);
        return spec;
    }

    bool valid_projection_boundary() const {
        return !relation.empty() && projection_only && !semantic_authority;
    }
};

struct VectorMultiSignalObjective {
    // L_total = α L_fold(N(k)) + β L_relation(A,r,B)
    //         + γ L_dynamics(F_Q, ΔF_GOOD)
    float alpha_fold = 1.0f;
    float beta_relation = 1.0f;
    float gamma_dynamics = 1.0f;

    bool uses_fold_topology = true;
    bool uses_relation_edges = true;
    bool uses_episode_dynamics = true;
    bool self_supervised = true;

    bool valid() const {
        return alpha_fold >= 0.0f && beta_relation >= 0.0f &&
               gamma_dynamics >= 0.0f &&
               uses_fold_topology && uses_relation_edges &&
               uses_episode_dynamics && self_supervised;
    }
};

enum class SemanticGeometryModel {
    FlatEmbedding,       // global vector fallback
    LocalTangentChart,   // N(k) -> T_k(M_mu)
    LocalMetricField,    // g(F, theta, r)
    GraphGeodesic        // typed path distance over B
};

inline const char* semantic_geometry_model_name(SemanticGeometryModel model) {
    switch (model) {
    case SemanticGeometryModel::FlatEmbedding:      return "flat_embedding";
    case SemanticGeometryModel::LocalTangentChart:  return "local_tangent_chart";
    case SemanticGeometryModel::LocalMetricField:   return "local_metric_field";
    case SemanticGeometryModel::GraphGeodesic:      return "graph_geodesic";
    default:                                        return "unknown";
    }
}

struct SemanticManifoldProjectionContract {
    // B is the authoritative discrete semantic topology. M_mu is a learned
    // geometry over B, not a replacement for B.
    std::string topology_symbol = "B";
    std::string learned_geometry_symbol = "M_mu";
    std::string local_chart_formula = "N(k)->T_k(M_mu)";
    std::string metric_formula = "g(F,theta,r)";
    std::string relevance_formula = "R_mu(Q,F,theta,r)";

    bool projection_only = true;
    bool semantic_authority = false;
    bool topology_authority = false;
    bool rebuildable_from_topology = true;
    bool phase_conditioned = true;
    bool relation_conditioned = true;
    bool delta_manifold_is_delta_topology = false;

    bool safe() const {
        return topology_symbol == "B" &&
               learned_geometry_symbol == "M_mu" &&
               projection_only && !semantic_authority && !topology_authority &&
               rebuildable_from_topology && phase_conditioned &&
               relation_conditioned && !delta_manifold_is_delta_topology;
    }
};

struct LocalSemanticMetricSpec {
    std::string phase = "Yax";
    std::string relation = "@related_to";
    SemanticGeometryModel model = SemanticGeometryModel::LocalMetricField;

    bool uses_local_tangent_chart = true;
    bool relation_conditioned = true;
    bool phase_conditioned = true;
    bool projection_only = true;
    bool semantic_authority = false;

    bool valid_projection_boundary() const {
        return !phase.empty() &&
               relation_geometry_kind_for(relation) != RelationGeometryKind::Unknown &&
               model == SemanticGeometryModel::LocalMetricField &&
               uses_local_tangent_chart && relation_conditioned &&
               phase_conditioned && projection_only && !semantic_authority;
    }
};

struct SemanticGeodesicProjectionSpec {
    std::string distance_symbol = "d_M(A,B)";
    std::string path_source = "B";
    SemanticGeometryModel model = SemanticGeometryModel::GraphGeodesic;

    bool typed_relation_costs = true;
    bool phase_conditioned = true;
    bool projection_only = true;
    bool semantic_authority = false;
    bool can_verify_relation = false;

    bool valid_projection_boundary() const {
        return distance_symbol == "d_M(A,B)" && path_source == "B" &&
               model == SemanticGeometryModel::GraphGeodesic &&
               typed_relation_costs && phase_conditioned && projection_only &&
               !semantic_authority && !can_verify_relation;
    }
};

struct VectorDecisionRiskSpec {
    // Statistical-decision view over projection rules:
    // R(theta,delta)=E_{x|theta}[Loss(theta,delta(x))]
    float prior_mass = 0.0f;
    float source_risk_bound = 1.0f;
    float target_risk_bound = 1.0f;
    float loss_cost = 0.0f;

    bool bayes_path = true;
    bool minimax_guard = true;
    bool projection_only = true;
    bool semantic_authority = false;
    bool topology_authority = false;

    bool unit_interval(float v) const {
        return v >= 0.0f && v <= 1.0f;
    }

    bool valid() const {
        return unit_interval(prior_mass) &&
               unit_interval(source_risk_bound) &&
               unit_interval(target_risk_bound) &&
               unit_interval(loss_cost) &&
               bayes_path && minimax_guard &&
               projection_only && !semantic_authority && !topology_authority;
    }

    bool inadmissible_minimax_rule() const {
        return valid() &&
               loss_cost > source_risk_bound &&
               loss_cost > target_risk_bound;
    }
};

enum class DifferentialMutationOp {
    Insert,
    Delete,
    Modify,
    Unchanged,
    Invalid
};

inline DifferentialMutationOp differential_mutation_op_for(std::string op) {
    std::transform(op.begin(), op.end(), op.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (op == "insert") return DifferentialMutationOp::Insert;
    if (op == "delete") return DifferentialMutationOp::Delete;
    if (op == "modify") return DifferentialMutationOp::Modify;
    if (op == "unchanged") return DifferentialMutationOp::Unchanged;
    return DifferentialMutationOp::Invalid;
}

struct DifferentialComparisonProjectionSpec {
    // ΔF_diff = < ΔM[Y,X], ΔRelations, R_minimax >
    DifferentialMutationOp mutation = DifferentialMutationOp::Unchanged;
    uint32_t base_version = 0;
    bool has_delta_cooccurrence = true;
    bool has_delta_relations = true;
    bool has_minimax_risk_bound = true;
    bool projection_only = true;
    bool can_commit_topology = false;

    bool valid_projection_boundary() const {
        return mutation != DifferentialMutationOp::Invalid &&
               has_delta_cooccurrence && has_delta_relations &&
               has_minimax_risk_bound && projection_only &&
               !can_commit_topology;
    }
};

enum class VectorProjectionAlignmentPolicy {
    SameGenerationOnly,
    OrthogonalProcrustes,
    AuthoritativeRegeneration
};

inline const char* vector_alignment_policy_name(VectorProjectionAlignmentPolicy p) {
    switch (p) {
    case VectorProjectionAlignmentPolicy::SameGenerationOnly:          return "same_generation_only";
    case VectorProjectionAlignmentPolicy::OrthogonalProcrustes:        return "orthogonal_procrustes";
    case VectorProjectionAlignmentPolicy::AuthoritativeRegeneration:   return "authoritative_regeneration";
    default:                                                           return "unknown";
    }
}

struct VectorProjectionGeneration {
    std::string projection_id = "vector:projection:bootstrap:v1:g0";
    uint32_t generation = 0;
    std::string learner_version = "bootstrap.deterministic_hash.v1";
    std::string topology_version = "runtime.semantic_containers";

    // ΔVector != ΔSemantics. A coordinate change is not a semantic change.
    bool delta_vector_is_semantic_delta = false;
    bool semantic_authority = false;
    bool rebuildable_from_authority = true;

    bool comparable_without_alignment(const VectorProjectionGeneration& other) const {
        return generation == other.generation &&
               projection_id == other.projection_id &&
               learner_version == other.learner_version &&
               topology_version == other.topology_version;
    }

    VectorProjectionAlignmentPolicy alignment_policy_for(
        const VectorProjectionGeneration& other) const
    {
        if (comparable_without_alignment(other))
            return VectorProjectionAlignmentPolicy::SameGenerationOnly;
        if (topology_version == other.topology_version)
            return VectorProjectionAlignmentPolicy::OrthogonalProcrustes;
        return VectorProjectionAlignmentPolicy::AuthoritativeRegeneration;
    }
};

struct VectorLearnerBoundary {
    bool projection_only = true;
    bool semantic_authority = false;
    bool can_verify_episode = false;
    bool can_change_verdict = false;
    bool can_commit_topology = false;
    bool can_advance_phase = false;
    bool can_mutate_fold = false;

    bool safe() const {
        return projection_only && !semantic_authority && !can_verify_episode &&
               !can_change_verdict && !can_commit_topology &&
               !can_advance_phase && !can_mutate_fold;
    }
};

// Semantic displacement extracted from one verified episode.
//
// Invariant: verdict must be "GOOD". FoldDelta cannot manufacture its own
// verification — the verdict is set by the extraction site from an external
// GOOD signal, never by the FoldDelta itself.
//
// ΔF = ⟨ΔM, ΔRelations, ΔW, ΔC⟩ — see episode.schema.json fold_delta section.
//
// For factory evolution episodes (factory_advisor_commit path):
//   cooccurrence : empty     — M updates come from full query/response cycles
//   relations    : populated — new capability and governs edges
//   weights      : delta_W=0 — propose_evolve does not change static_weight
//   confidence   : delta_C=0 — propose_evolve does not change confidence
struct FoldDelta {
    CooccurrenceDelta  cooccurrence;  // P_M(ΔF)
    RelationDelta      relations;     // P_R(ΔF)
    WeightDelta        weights;       // P_W(ΔF)
    ConfidenceDelta    confidence;    // P_C(ΔF)

    // Provenance — must survive PrincipalDirection compression.
    // jrom_seq links back to the raw episode journal entry so Ch'en can
    // answer "why does this direction exist?" by walking FoldDirection → ΔF → Episode.
    std::string  micronaut_id;
    uint64_t     jrom_seq    = 0;   // sequence number returned by jrom.log()
    std::string  extracted_at;      // ISO-8601 timestamp set at extraction time
    std::string  verdict;           // "GOOD" for valid deltas

    // VECTOR-µ consumes ΔF as evidence, not authority.
    // A projected vector delta may be useful for retrieval, but it is not ΔF.
    VectorLearnerBoundary vector_boundary;

    bool is_empty() const {
        return cooccurrence.empty() && relations.empty()
            && weights.empty() && confidence.empty();
    }

    bool vector_projection_allowed() const {
        return verdict == "GOOD" && vector_boundary.safe();
    }
};

// Sentinel: non-GOOD verdict → no delta produced.
inline FoldDelta no_delta() { return FoldDelta{}; }

} // namespace khanary::kuhul
