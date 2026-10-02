"""Validate the VECTOR-µ provider registry in vectors.xml.

Suite name: vectors_xml  (must match run_conformance.py required set)
"""
from pathlib import Path
import json
import re
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).parent))
from conformance_common import verdict


ROOT = Path(__file__).resolve().parents[1]


def _bool_attr(elem, name, default=False):
    value = elem.attrib.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes"}


def _gml_stats(path):
    text = path.read_text(encoding="utf-8")
    nodes = len(re.findall(r"(?m)^\s*node\s*\[", text))
    edges = len(re.findall(r"(?m)^\s*edge\s*\[", text))
    match = re.search(r"(?m)^\s*directed\s+([01])", text)
    directed = match.group(1) if match else None
    return nodes, edges, directed


def main():
    cases = []
    try:
        path = ROOT / "vectors.xml"
        tree = ET.parse(path)
        root = tree.getroot()
        cases.append(("root_vector_registry", root.tag == "vector_registry"))
        cases.append(("projection_only_authority",
                      root.attrib.get("authority") == "projection-only" and
                      root.attrib.get("semantic_authority") == "false"))

        providers = {p.attrib.get("id"): p for p in root.findall(".//provider")}
        required_providers = {
            "gml-topology",
            "field-node-vectors",
            "projection-abi",
            "gyro-abi",
            "gml-validator-fixtures",
            "dml-ops",
            "birdsong-stb",
            "klsl-mesh-compression",
        }
        cases.append(("required_local_providers_present",
                      required_providers.issubset(providers)))

        local_missing = []
        for elem in (list(root.findall(".//provider")) +
                     list(root.findall(".//sidecar")) +
                     list(root.findall(".//model")) +
                     list(root.findall(".//header")) +
                     list(root.findall(".//topology")) +
                     list(root.findall(".//grammar"))):
            if not _bool_attr(elem, "require"):
                continue
            source = elem.attrib.get("source", "")
            if not source:
                local_missing.append("<missing-source>")
                continue
            if not (ROOT / source).exists():
                local_missing.append(source)
        cases.append(("required_local_sources_exist", not local_missing))

        corpus = root.find(".//gml_topology_corpus")
        cases.append(("gml_topology_corpus_declared", corpus is not None))
        gml_counts_ok = True
        if corpus is not None:
            for topo in corpus.findall("topology"):
                source = topo.attrib.get("source")
                if not source:
                    gml_counts_ok = False
                    continue
                try:
                    nodes, edges, directed = _gml_stats(ROOT / source)
                    gml_counts_ok = (
                        gml_counts_ok and
                        nodes == int(topo.attrib.get("nodes", "-1")) and
                        edges == int(topo.attrib.get("edges", "-1")) and
                        directed == topo.attrib.get("directed")
                    )
                except (OSError, ValueError):
                    gml_counts_ok = False
        cases.append(("gml_topology_counts_match", gml_counts_ok))

        projection_abi = json.loads((ROOT / "data/projection_abi_v1.json").read_text(encoding="utf-8"))
        gyro_abi = json.loads((ROOT / "data/gyro_abi_v1.json").read_text(encoding="utf-8"))
        cases.append(("projection_abi_declared", "⟁PROJECTION_ABI_v1" in projection_abi))
        cases.append(("gyro_abi_declared", "⟁GYRO_ABI_v1" in gyro_abi))

        fixtures = json.loads((ROOT / "data/conformance/gml_validator.json").read_text(encoding="utf-8"))
        cases.append(("gml_validator_fixtures_present",
                      isinstance(fixtures, list) and len(fixtures) >= 10 and
                      all("id" in row and "gml" in row and "grade" in row for row in fixtures)))

        grammar_contracts = {g.attrib.get("id"): g for g in root.findall(".//grammar")}
        cases.append(("grammar_contracts_declared",
                      {"gml-ebnf", "gml-validator-contract", "fold-content-dictionary",
                       "vector-mathml-foldpair", "fold-graph-laws"}.issubset(grammar_contracts)))

        gml_ebnf = (ROOT / "data/grammar/gml.khanary.ebnf").read_text(encoding="utf-8")
        cases.append(("gml_ebnf_declares_relation_overlay",
                      "relation_verb" in gml_ebnf and
                      "gml_overlay_id" in gml_ebnf and
                      "@contradicts" in gml_ebnf and
                      "@causes" in gml_ebnf))
        cases.append(("gml_ebnf_declares_continuous_field_attrs",
                      all(token in gml_ebnf for token in (
                          "boundary_type", "vertex_pot", "phase_shift",
                          "divergence", "singularity", "text_boundary",
                          "curve_id", "parameter_t", "arc_length", "curvature",
                          "scalar_potential", "field_kind", "curl_x",
                          "provider_id", "parameter_id", "parameter_generation",
                          "training_step", "loss_value", "grad_norm",
                          "learning_rate", "gradient_clip",
                           "local_entropy", "entropy_spike", "entropy_bound",
                           "eh_x", "eh_y", "eh_z", "diffusion_beta_inv",
                           "geodesic_entropy", "triangular_scale",
                           "triangle_id", "simplex_weight",
                           "length", "potential", "refractive",
                           "wave_number", "frequency", "amplitude_re",
                           "vertex_bc", "index_type",
                           "tensor_rank_p", "tensor_rank_q", "is_smooth_section",
                           "riemann_curvature_scalar", "jacobian_determinant",
                           "g_00", "g_01", "g_10", "g_11", "jacobian_w",
                           "tensor_field"))))

        gml_validator = json.loads((ROOT / "data/grammar/gml.validator.json").read_text(encoding="utf-8"))
        entropic_schema = json.loads((ROOT / "data/schema/entropic-well.schema.json").read_text(encoding="utf-8"))
        entropic_wells = json.loads((ROOT / "data/world/entropic_wells.json").read_text(encoding="utf-8"))
        well_props = entropic_schema.get("$defs", {}).get("EntropicWell", {}).get("properties", {})
        mesh_schema = well_props.get("inscribed_mesh_geometry", {})
        cases.append(("entropic_well_v4_declares_tensor_bundle_geometry",
                      entropic_schema.get("$id") == "urn:kuhul:khanary:schema:entropic-well:v4" and
                      entropic_wells.get("$schema") == "urn:kuhul:khanary:schema:entropic-well:v4" and
                      "Tensor_Bundle_Manifold" in
                      entropic_schema.get("properties", {}).get("metric_context", {}).get("enum", []) and
                      {"tensor_field_type", "local_metric_tensor_g_ij"}.issubset(well_props.keys()) and
                      {"circumradius_R", "inradius_r", "circumcenter_coord",
                       "triangular_entropy_h_delta", "riemann_curvature_scalar"}.issubset(
                          mesh_schema.get("properties", {}).keys()) and
                      "jacobian_determinant" in
                      well_props.get("field_warp", {}).get("properties", {}) and
                      "euler_triangle_bound" in
                      entropic_schema.get("properties", {}).get("laws", {}).get("properties", {}) and
                      "tensor_coordinate_transform" in
                      entropic_schema.get("properties", {}).get("laws", {}).get("properties", {})))
        euler_ok = True
        for well in entropic_wells.get("active_wells", []):
            mesh = well.get("inscribed_mesh_geometry")
            if not mesh:
                euler_ok = False
                continue
            center = mesh.get("circumcenter_coord")
            try:
                R = float(mesh["circumradius_R"])
                r = float(mesh["inradius_r"])
                h = float(mesh["triangular_entropy_h_delta"])
                euler_ok = (
                    euler_ok and R > 0.0 and r > 0.0 and h >= 0.0 and
                    R + 1.0e-9 >= 2.0 * r and
                    isinstance(center, list) and len(center) == 3 and
                    all(isinstance(v, (int, float)) for v in center)
                )
            except (KeyError, TypeError, ValueError):
                euler_ok = False
        cases.append(("entropic_wells_satisfy_euler_triangle_inequality", euler_ok))
        tensor_wells_ok = True
        for well in entropic_wells.get("active_wells", []):
            try:
                tf = well["tensor_field_type"]
                metric = well["local_metric_tensor_g_ij"]
                warp = well["field_warp"]
                mesh = well["inscribed_mesh_geometry"]
                p_rank = int(tf["contravariant_rank_p"])
                q_rank = int(tf["covariant_rank_q"])
                g00 = float(metric["g00"])
                g01 = float(metric["g01"])
                g10 = float(metric["g10"])
                g11 = float(metric["g11"])
                det_g = (g00 * g11) - (g01 * g10)
                det_j = float(warp["jacobian_determinant"])
                float(mesh["riemann_curvature_scalar"])
                tensor_wells_ok = (
                    tensor_wells_ok and p_rank >= 0 and q_rank >= 0 and
                    tf.get("is_smooth_section") is True and
                    abs(g01 - g10) <= 1.0e-9 and abs(det_g) > 1.0e-9 and
                    abs(det_j) > 1.0e-9
                )
            except (KeyError, TypeError, ValueError):
                tensor_wells_ok = False
        cases.append(("entropic_wells_satisfy_tensor_bundle_bounds", tensor_wells_ok))

        pipeline_stages = [row.get("stage") for row in gml_validator.get("pipeline", [])]
        cases.append(("gml_validator_declares_admission_pipeline",
                      gml_validator.get("id") == "gml.validator.v1" and
                      gml_validator.get("kind") == "gml_semantic_validator" and
                      pipeline_stages == ["ParseGML", "ValidateGML", "Proposal",
                                          "Ch'en", "XCFE", "XulGate"]))
        checks = gml_validator.get("validation_checks", {})
        cases.append(("gml_validator_declares_relation_candidate_law",
                      "R5" in checks and
                      "infer_relation_from_lexical(x) ≠ VerifiedRelation(x)" in
                      checks["R5"].get("formal", "")))
        cases.append(("gml_validator_declares_decision_theory_laws",
                      {"S1", "S2", "S3", "S4"}.issubset(checks) and
                      gml_validator.get("decision_theory", {}).get("minimax", "").startswith(
                          "minimize maximum structural loss") and
                      checks["S4"].get("error") == "INADMISSIBLE_VECTOR_RULE"))
        cases.append(("gml_validator_declares_differential_graph_laws",
                      {"D1", "D2", "D3", "D4"}.issubset(checks) and
                      gml_validator.get("differential_comparison", {}).get("delta_graph", "").startswith(
                          "ΔG=") and
                      checks["D1"].get("error") == "INVALID_MUTATION_OP"))
        cases.append(("gml_validator_declares_continuous_field_laws",
                      {"Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8",
                       "Q9", "Q10", "Q11", "Q12", "Q13", "Q14"}.issubset(checks) and
                      gml_validator.get("quantum_graph", {}).get("operator", "").startswith("GraphHamiltonian") and
                      gml_validator.get("plane_wave_field", {}).get("edge_wave", "").startswith("psi_e") and
                      gml_validator.get("continuous_vector_field", {}).get("lie_bracket", "").startswith("[X,Y]") and
                      gml_validator.get("structural_boundaries", {}).get("role", "").startswith("Boundary tokens") and
                      gml_validator.get("differentiable_curve", {}).get("curve", "").startswith("gamma:I->X") and
                      gml_validator.get("euclidean_gradient_field", {}).get("zero_curl", "").startswith("curl") and
                      checks["Q14"].get("error") == "EUCLIDEAN_CURL_NONZERO"))
        cases.append(("gml_validator_declares_training_gradient_laws",
                      {"T1", "T2", "T3", "T4", "T5", "T6", "T7"}.issubset(checks) and
                      gml_validator.get("training_gradients", {}).get("gradient", "").startswith("g_phi") and
                      checks["T5"].get("error") == "LEARNING_RATE_NONPOSITIVE" and
                      "TrainingGradient != FoldDelta" in
                      gml_validator.get("training_gradients", {}).get("authority_boundary", "")))
        cases.append(("gml_validator_declares_entropy_gradient_laws",
                      {"E1", "E2", "E3", "E4", "E5", "E6", "E7"}.issubset(checks) and
                      gml_validator.get("entropy_gradient", {}).get("entropy", "").startswith("H(x)=") and
                      gml_validator.get("free_energy_flow", {}).get("flow", "").startswith("V(x)=") and
                      checks["E4"].get("error") == "ENTROPY_BOUND_EXCEEDED"))
        cases.append(("gml_validator_declares_tensor_bundle_laws",
                      {"TB1", "TB2", "TB3", "TB4", "TB5", "TB6", "TB7"}.issubset(checks) and
                      gml_validator.get("tensor_bundle", {}).get("component") == "TENSOR-µ" and
                      gml_validator.get("tensor_bundle", {}).get("field_kind") == "tensor_field" and
                      "cannot verify semantic truth" in
                      gml_validator.get("tensor_bundle", {}).get("authority_boundary", "")))
        cases.append(("gml_validator_declares_lorentz_transition_laws",
                      {"LB1", "LB2", "LB3"}.issubset(checks) and
                      checks["LB2"].get("error") == [
                          "LORENTZ_ENABLED_NOT_BOOL",
                          "LORENTZ_ALIGNMENT_OUT_OF_RANGE",
                          "LORENTZ_MASS_SCALE_NEGATIVE"
                      ] and
                      checks["LB3"].get("error") == "LORENTZ_NOT_IDENTITY_WHEN_DISABLED"))
        cases.append(("gml_validator_declares_deductive_system_laws",
                      {"L1", "L2", "L3"}.issubset(checks) and
                      checks["L1"].get("error") == "DEDUCTIVE_PROOF_REQUIRED" and
                      checks["L2"].get("error") == "MISS_COLLAPSED_TO_CONTRADICTION" and
                      checks["L3"].get("error") == "AXIOM_EXTENSION_CONTRADICTS_LAW" and
                      gml_validator.get("deductive_system", {}).get("formal_system") ==
                      "D=<Lambda,Sigma,{(Gamma,phi)}>" and
                      "UNKNOWN and MISS" in
                      gml_validator.get("deductive_system", {}).get("axiomatic_incompleteness_boundary", "")))
        cases.append(("gml_validator_declares_hyperbolic_event_laws",
                      {"HE1", "HE2", "HE3", "HE4", "HE5", "HE6", "HE7"}.issubset(checks) and
                      checks["HE2"].get("error") == "RAPIDITY_RHO_NEGATIVE" and
                      "MINKOWSKI_INTERVAL_CLASS_MISMATCH" in checks["HE4"].get("error", []) and
                      gml_validator.get("hyperbolic_key_grammar_event", {}).get("schema") ==
                      "urn:kuhul:khanary:schema:key-grammar-event:v1" and
                      gml_validator.get("hyperbolic_key_grammar_event", {}).get("distance") ==
                      "d(u,v)=arcosh(-B(u,v))" and
                      "semantic_authority remains false" in
                      gml_validator.get("hyperbolic_key_grammar_event", {}).get("authority_boundary", "")))

        om_ns = {"om": "http://www.openmath.org/OpenMathCD"}
        fold_cd = ET.parse(ROOT / "data/grammar/fold.cd.xml").getroot()
        cd_uri = fold_cd.findtext("om:CDURI", default="", namespaces=om_ns)
        symbols = {elem.findtext("om:Name", default="", namespaces=om_ns)
                   for elem in fold_cd.findall("om:CDDefinition", om_ns)}
        cases.append(("fold_cd_declares_symbols",
                      cd_uri == "urn:kuhul:fold-graph:v1" and
                      {"FoldPair", "FoldPairLower", "Axis", "supp", "setbuilder",
                       "PriorDistribution", "Probability", "Loss", "Risk",
                       "DecisionRule", "BayesRisk", "MinimaxRule",
                       "MinimaxRiskBound", "DeltaFieldDiff",
                       "DeltaCooccurrenceMass", "DeltaRelations",
                       "GraphHamiltonian", "IncidentEdges", "DerivativeAtVertex",
                       "FluxConservation", "PlaneWaveSuperposition",
                       "ScatteringMatrix", "KroneckerDelta", "Degree",
                       "FieldZeroes", "LocalIndex", "EulerCharacteristic",
                       "GradientFlow", "LieBracket", "ConditionalProbability",
                       "DifferentiableCurve", "TangentVelocity", "SemanticDisplacement",
                       "TrajectoryLineIntegral", "CurveCurvature", "ScalarPotential",
                       "EuclideanGradient", "ConservativeLineIntegral",
                       "ZeroCurlConstraint", "LieDerivative",
                       "ProviderParameters", "ProjectionLoss", "TrainingGradient",
                       "GradientNorm", "GradientClip", "ParameterUpdate",
                       "ShannonEntropy", "EntropyGradient", "EntropySpike",
                       "EntropyBound", "FreeEnergyFlow", "FokkerPlanckFlow",
                        "DiffusionScale", "Circumradius", "Inradius",
                        "Circumcenter", "EulerTriangleInequality",
                        "StructuralBoundaryReset", "TensorField", "TensorRank",
                        "MetricTensor", "JacobianMatrix", "InverseJacobian",
                        "TensorTransform", "SmoothSection", "TensorBundle",
                        "CoordinateTransition", "LogicalConsequence",
                        "DeductiveProofTrace", "MissState",
                        "MinkowskiBilinearForm", "HyperbolicDistance",
                        "HyperbolicEvent", "Rapidity",
                        "MinkowskiIntervalClass", "EinsteinContraction",
                        "LorentzFrameTransition", "LorentzBoostTensor",
                        "SemanticSignature", "Neighborhood", "AlgebraicFold",
                        "XulCollapse", "SignatureIsomorphism",
                        "MatrixRank"}.issubset(symbols)))

        mml_ns = {"mml": "http://www.w3.org/1998/Math/MathML"}
        vectors_mml = ET.parse(ROOT / "data/grammar/vectors.mml").getroot()
        annotations = "\n".join(
            (elem.text or "")
            for elem in vectors_mml.findall(".//mml:annotation", mml_ns)
        )
        csymbols = {
            (elem.attrib.get("cd"), elem.text or "")
            for elem in vectors_mml.findall(".//mml:csymbol", mml_ns)
        }
        cases.append(("vectors_mml_declares_foldpair_contract",
                      vectors_mml.tag == "{http://www.w3.org/1998/Math/MathML}math" and
                      vectors_mml.attrib.get("display") == "block" and
                      ("urn:kuhul:fold-graph:v1", "FoldPair") in csymbols and
                      "FoldPair(S) = < M[Y,X], C[Y,X]^T >" in annotations and
                      "shared semantic axis: X" in annotations))
        cases.append(("vectors_mml_separates_mass_from_wcr_weight",
                      "M != W_Y" in annotations and
                      "M = co-occurrence mass" in annotations and
                      "M[Y][X] += 1   # co-occurrence mass; not the WCR W" in annotations))
        mml_text = (ROOT / "data/grammar/vectors.mml").read_text(encoding="utf-8")
        cases.append(("vectors_mml_declares_no_pattern_inflation",
                      "no-pattern-inflation" in mml_text and
                      "|supp M[Y,·]| = #{X ∈ S : Y ∈ C_X}" in mml_text))
        cases.append(("vectors_mml_declares_decision_risk",
                      ("urn:kuhul:fold-graph:v1", "Risk") in csymbols and
                      ("urn:kuhul:fold-graph:v1", "Probability") in csymbols and
                      ("urn:kuhul:fold-graph:v1", "Loss") in csymbols and
                      ("urn:kuhul:fold-graph:v1", "DecisionRule") in csymbols and
                      "Risk(theta, delta) = sum_{x in S}" in annotations and
                      "risk constrains representation geometry only" in annotations))
        cases.append(("vectors_mml_declares_differential_risk",
                      ("urn:kuhul:fold-graph:v1", "DeltaFieldDiff") in csymbols and
                      ("urn:kuhul:fold-graph:v1", "DeltaCooccurrenceMass") in csymbols and
                      ("urn:kuhul:fold-graph:v1", "DeltaRelations") in csymbols and
                      ("urn:kuhul:fold-graph:v1", "MinimaxRiskBound") in csymbols and
                      "DeltaF_diff = < DeltaM[Y,X], DeltaRelations, R_minimax >" in annotations and
                      "cannot commit B" in annotations))
        cases.append(("vectors_mml_declares_quantum_graph_fields",
                      ("urn:kuhul:fold-graph:v1", "GraphHamiltonian") in csymbols and
                      ("urn:kuhul:fold-graph:v1", "FluxConservation") in csymbols and
                      ("urn:kuhul:fold-graph:v1", "IncidentEdges") in csymbols and
                      ("urn:kuhul:fold-graph:v1", "DerivativeAtVertex") in csymbols and
                      "GraphHamiltonian: H psi_e(x)" in annotations and
                      "FluxConservation(v):" in annotations))
        cases.append(("vectors_mml_declares_wave_scattering_flow",
                      ("urn:kuhul:fold-graph:v1", "PlaneWaveSuperposition") in csymbols and
                      ("urn:kuhul:fold-graph:v1", "ScatteringMatrix") in csymbols and
                      ("urn:kuhul:fold-graph:v1", "GradientFlow") in csymbols and
                      ("urn:kuhul:fold-graph:v1", "LieBracket") in csymbols and
                      "PlaneWaveSuperposition: psi_e(x)" in annotations and
                      "ScatteringMatrix(v,m,j)" in annotations and
                      "LieBracket(X,Y,f)" in annotations))
        cases.append(("vectors_mml_declares_boundary_reset",
                      ("urn:kuhul:fold-graph:v1", "FieldZeroes") in csymbols and
                      ("urn:kuhul:fold-graph:v1", "LocalIndex") in csymbols and
                      ("urn:kuhul:fold-graph:v1", "EulerCharacteristic") in csymbols and
                      ("urn:kuhul:fold-graph:v1", "ConditionalProbability") in csymbols and
                      ("urn:kuhul:fold-graph:v1", "StructuralBoundaryReset") in csymbols and
                      "Poincare-Hopf:" in annotations and
                      "N-gram windows must not bleed across structural sinks" in annotations))
        cases.append(("vectors_mml_declares_curve_gradient_contract",
                      ("urn:kuhul:fold-graph:v1", "DifferentiableCurve") in csymbols and
                      ("urn:kuhul:fold-graph:v1", "TangentVelocity") in csymbols and
                      ("urn:kuhul:fold-graph:v1", "SemanticDisplacement") in csymbols and
                      ("urn:kuhul:fold-graph:v1", "TrajectoryLineIntegral") in csymbols and
                      ("urn:kuhul:fold-graph:v1", "CurveCurvature") in csymbols and
                      ("urn:kuhul:fold-graph:v1", "EuclideanGradient") in csymbols and
                      ("urn:kuhul:fold-graph:v1", "ConservativeLineIntegral") in csymbols and
                      ("urn:kuhul:fold-graph:v1", "ScalarPotential") in csymbols and
                      ("urn:kuhul:fold-graph:v1", "ZeroCurlConstraint") in csymbols and
                      "DifferentiableCurve: gamma:I->X" in annotations and
                      "ZeroCurlConstraint: partial V_i" in annotations))
        cases.append(("vectors_mml_declares_training_gradient_contract",
                      ("urn:kuhul:fold-graph:v1", "TrainingGradient") in csymbols and
                      ("urn:kuhul:fold-graph:v1", "ProjectionLoss") in csymbols and
                      ("urn:kuhul:fold-graph:v1", "ProviderParameters") in csymbols and
                      ("urn:kuhul:fold-graph:v1", "ParameterUpdate") in csymbols and
                      ("urn:kuhul:fold-graph:v1", "GradientClip") in csymbols and
                      ("urn:kuhul:fold-graph:v1", "GradientNorm") in csymbols and
                      "TrainingGradient: g_phi = grad_phi L_total" in annotations and
                      "ParameterUpdate: phi_next = phi - eta" in annotations and
                      "GradientNorm: grad_norm = ||g_phi||" in annotations))
        cases.append(("vectors_mml_declares_entropy_gradient_contract",
                      ("urn:kuhul:fold-graph:v1", "ShannonEntropy") in csymbols and
                      ("urn:kuhul:fold-graph:v1", "EntropyGradient") in csymbols and
                      ("urn:kuhul:fold-graph:v1", "FreeEnergyFlow") in csymbols and
                      ("urn:kuhul:fold-graph:v1", "FokkerPlanckFlow") in csymbols and
                      ("urn:kuhul:fold-graph:v1", "DiffusionScale") in csymbols and
                      "ShannonEntropy: H(x)" in annotations and
                      "EntropyGradient: grad H(x)" in annotations and
                      "FreeEnergyFlow: V(x)" in annotations and
                      "FokkerPlanckFlow: partial_t rho" in annotations))
        cases.append(("vectors_mml_declares_inscribed_mesh_boundary",
                      ("urn:kuhul:fold-graph:v1", "Circumradius") in csymbols and
                      ("urn:kuhul:fold-graph:v1", "Inradius") in csymbols and
                      "EulerTriangleInequality: R_circum >= 2 * r_in" in annotations and
                      "cannot mutate semantic topology B" in annotations))
        cases.append(("vectors_mml_declares_tensor_transform_contract",
                      ("urn:kuhul:fold-graph:v1", "TensorTransform") in csymbols and
                      ("urn:kuhul:fold-graph:v1", "TensorField") in csymbols and
                      ("urn:kuhul:fold-graph:v1", "JacobianMatrix") in csymbols and
                      ("urn:kuhul:fold-graph:v1", "InverseJacobian") in csymbols and
                      "TensorTransform: T^{i'}_{j'}" in annotations and
                      "TENSOR-µ tensor-bundle geometry is projection-only" in annotations))
        cases.append(("vectors_mml_declares_deductive_system_contract",
                      ("urn:kuhul:fold-graph:v1", "LogicalConsequence") in csymbols and
                      ("urn:kuhul:fold-graph:v1", "DeductiveProofTrace") in csymbols and
                      ("urn:kuhul:fold-graph:v1", "MissState") in csymbols and
                      "SemanticCompletenessBoundary:" in annotations and
                      "AxiomaticIncompletenessBoundary:" in annotations and
                      "MISS is not CONTRADICTED" in annotations))
        cases.append(("vectors_mml_declares_hyperbolic_event_contract",
                      ("urn:kuhul:fold-graph:v1", "MinkowskiBilinearForm") in csymbols and
                      ("urn:kuhul:fold-graph:v1", "HyperbolicDistance") in csymbols and
                      "MinkowskiBilinearForm: B(u,v)" in annotations and
                      "HyperbolicEventDistance: d(u,v) = arcosh(-B(u,v))" in annotations and
                      "interval_class gates routing but never verifies facts" in annotations))
        cases.append(("vectors_mml_declares_einstein_contraction_contract",
                      ("urn:kuhul:fold-graph:v1", "EinsteinContraction") in csymbols and
                      ("urn:kuhul:fold-graph:v1", "LorentzFrameTransition") in csymbols and
                      ("urn:kuhul:fold-graph:v1", "LorentzBoostTensor") in csymbols and
                      "EinsteinContraction: u^i = L^i_j v^j" in annotations and
                      "LorentzBoostTensor(rho,n)" in annotations and
                      "dot(boost_row_i, v)" in annotations and
                      "cannot establish semantic authority" in annotations))
        cases.append(("vectors_mml_declares_catamorphic_fold_contract",
                      ("urn:kuhul:fold-graph:v1", "XulCollapse") in csymbols and
                      ("urn:kuhul:fold-graph:v1", "AlgebraicFold") in csymbols and
                      ("urn:kuhul:fold-graph:v1", "Neighborhood") in csymbols and
                      "CatamorphicFoldLoop: Xul(Fold_alpha(N(k), theta)) -> k_prime" in annotations and
                      "invariant signature F" in annotations and
                      "SignatureIsomorphism(F_base,F_transformed)" in annotations and
                      "cannot mutate FoldGraph B" in annotations))

        key_grammar_ebnf = (ROOT / "data/grammar/key-grammar-json.runtime.ebnf").read_text(encoding="utf-8")
        cases.append(("key_grammar_declares_hyperbolic_event_schema",
                      "hyperbolic_event_row" in key_grammar_ebnf and
                      "urn:kuhul:khanary:schema:key-grammar-event:v1" in key_grammar_ebnf and
                      "hyperbolic_metrics" in key_grammar_ebnf and
                      "Minkowski_Minus_Plus_Plus_Plus" in key_grammar_ebnf and
                      "q_he_semantic_authority" in key_grammar_ebnf and
                      '"false"' in key_grammar_ebnf))

        kfg_ns = "{urn:kuhul:fold-graph:v1}"
        fold_graph = ET.parse(ROOT / "data/grammar/fold_graph.xml").getroot()
        law_ids = {elem.attrib.get("id") for elem in fold_graph.findall(f".//{kfg_ns}law")}
        learning_invariant = fold_graph.find(f"{kfg_ns}learning-invariant") is not None
        cases.append(("fold_graph_declares_c6_topology_laws",
                      fold_graph.tag == f"{kfg_ns}fold-graph" and
                      fold_graph.attrib.get("topology-symbol") == "B" and
                      fold_graph.attrib.get("cycle") == "Pop Wo Yax Sek Ch'en Xul" and
                      learning_invariant and
                      {"fold-not-xul", "unfold-expands", "topology-preservation",
                       "representation-not-truth", "fold-delta-not-vec-delta",
                       "apply-field-not-commit-topology"}.issubset(law_ids)))

        dml_cpp = (ROOT / "src/gpu/dml_ops.cpp").read_text(encoding="utf-8")
        dml_h = (ROOT / "src/gpu/dml_ops.h").read_text(encoding="utf-8")
        cases.append(("dml_ops_declares_vector_gate",
                      "semantic_attention_gate" in dml_cpp and
                      "semantic_attention_gate" in dml_h))
        cases.append(("dml_ops_declares_projection_ops",
                      all(token in dml_h for token in ("gemm", "softmax", "reduce_sum", "attention"))))

        gml = providers.get("gml-topology")
        cases.append(("gml_is_topology_authority",
                      gml is not None and
                      gml.attrib.get("topology_authority") == "true" and
                      gml.attrib.get("semantic_authority") == "false"))

        projection_only = all(
            p.attrib.get("semantic_authority") == "false"
            for p in providers.values()
            if p.attrib.get("id") != "gml-topology"
        )
        cases.append(("providers_projection_only", projection_only))

        invariants = {i.attrib.get("id"): i.attrib.get("value")
                      for i in root.findall(".//invariant")}
        cases.append(("similarity_relatedness_boundary",
                      invariants.get("similarity_subset_relatedness") == "true"))
        cases.append(("gml_before_geometry_boundary",
                      invariants.get("gml_before_geometry") == "true"))
        cases.append(("projection_cannot_redefine_semantics",
                      invariants.get("projection_cannot_redefine_semantics") == "true"))
        cases.append(("semantic_topology_not_learned_manifold",
                      invariants.get("semantic_topology_not_learned_manifold") == "true"))
        cases.append(("metric_field_not_semantic_authority",
                      invariants.get("metric_field_not_semantic_authority") == "true"))
        cases.append(("quantum_graph_projection_boundary",
                      invariants.get("quantum_graph_not_semantic_authority") == "true"))
        cases.append(("structural_boundary_ngram_reset",
                      invariants.get("structural_boundaries_reset_ngrams") == "true"))
        cases.append(("differentiable_curve_projection_boundary",
                      invariants.get("differentiable_curve_projection_boundary") == "true"))
        cases.append(("euclidean_gradient_projection_only",
                      invariants.get("euclidean_gradient_projection_only") == "true"))
        cases.append(("training_gradient_not_fold_delta",
                      invariants.get("training_gradient_not_fold_delta") == "true"))
        cases.append(("entropy_gradient_projection_only",
                      invariants.get("entropy_gradient_projection_only") == "true"))
        cases.append(("inscribed_mesh_projection_only",
                      invariants.get("inscribed_mesh_projection_only") == "true"))
        cases.append(("tensor_field_projection_only",
                      invariants.get("tensor_field_projection_only") == "true"))
        cases.append(("vector_mu_rank1_provider",
                      invariants.get("vector_mu_rank1_provider") == "true"))

        manifold = root.find(".//semantic_manifold_contract")
        manifold_laws = {
            law.attrib.get("id"): (law.text or "")
            for law in root.findall(".//semantic_manifold_contract/law")
        }
        cases.append(("semantic_manifold_contract_declared",
                      manifold is not None and
                      manifold.attrib.get("id") == "vector:semantic_manifold:v1" and
                      manifold.attrib.get("owner") == "TENSOR-µ" and
                      manifold.attrib.get("legacy_owner") == "VECTOR-µ" and
                      manifold.attrib.get("semantic_authority") == "false" and
                      manifold.attrib.get("topology_authority") == "false" and
                      manifold.attrib.get("rebuildable") == "true"))
        if manifold is not None:
            topology = manifold.find("topology")
            geometry = manifold.find("learned_geometry")
            chart = manifold.find("local_chart")
            metric = manifold.find("metric")
            distance = manifold.find("geodesic_distance")
        else:
            topology = geometry = chart = metric = distance = None
        cases.append(("semantic_manifold_b_over_mmu_boundary",
                      topology is not None and topology.attrib.get("symbol") == "B" and
                      geometry is not None and geometry.attrib.get("symbol") == "M_mu" and
                      geometry.attrib.get("over") == "B" and
                      geometry.attrib.get("projection_only") == "true"))
        cases.append(("semantic_manifold_local_metric_declared",
                      chart is not None and chart.attrib.get("source") == "N(k)" and
                      chart.attrib.get("target") == "T_k(M_mu)" and
                      metric is not None and metric.attrib.get("symbol") == "g(F,theta,r)" and
                      distance is not None and distance.attrib.get("symbol") == "d_M(A,B)"))
        cases.append(("semantic_manifold_delta_law_declared",
                      "Delta(M_mu) does not imply Delta(B)." in
                      manifold_laws.get("delta_manifold_not_delta_topology", "") and
                      "R_mu(Q,F,theta,r)" in
                      manifold_laws.get("phase_conditioned_relevance", "")))

        decision = root.find(".//decision_theory_contract")
        decision_laws = {
            law.attrib.get("id"): (law.text or "")
            for law in root.findall(".//decision_theory_contract/law")
        }
        cases.append(("decision_theory_contract_declared",
                      decision is not None and
                      decision.attrib.get("id") == "vector:decision_theory:v1" and
                      decision.attrib.get("owner") == "TENSOR-µ" and
                      decision.attrib.get("legacy_owner") == "VECTOR-µ" and
                      decision.attrib.get("semantic_authority") == "false" and
                      decision.attrib.get("topology_authority") == "false" and
                      decision.attrib.get("projection_only") == "true"))
        if decision is not None:
            prior = decision.find("prior")
            loss = decision.find("loss")
            risk = decision.find("risk")
            minimax = decision.find("minimax")
            diff = decision.find("differential")
        else:
            prior = loss = risk = minimax = diff = None
        cases.append(("decision_theory_prior_loss_risk_declared",
                      prior is not None and prior.attrib.get("gml_attr") == "prior_mass" and
                      loss is not None and loss.attrib.get("gml_attr") == "loss_cost" and
                      risk is not None and risk.attrib.get("symbol") == "R(theta,delta)" and
                      risk.attrib.get("mathml") == "data/grammar/vectors.mml"))
        cases.append(("decision_theory_minimax_diff_declared",
                      minimax is not None and
                      minimax.attrib.get("error") == "INADMISSIBLE_VECTOR_RULE" and
                      diff is not None and diff.attrib.get("symbol") == "DeltaF_diff" and
                      "does not verify relations or commit B" in
                      decision_laws.get("risk-not-truth", "")))

        field = root.find(".//continuous_field_contract")
        field_laws = {
            law.attrib.get("id"): (law.text or "")
            for law in root.findall(".//continuous_field_contract/law")
        }
        cases.append(("continuous_field_contract_declared",
                      field is not None and
                      field.attrib.get("id") == "vector:continuous_field:v1" and
                      field.attrib.get("owner") == "TENSOR-µ" and
                      field.attrib.get("legacy_owner") == "VECTOR-µ" and
                      field.attrib.get("semantic_authority") == "false" and
                      field.attrib.get("topology_authority") == "false" and
                      field.attrib.get("projection_only") == "true"))
        if field is not None:
            hamiltonian = field.find("hamiltonian")
            plane_wave = field.find("plane_wave")
            flow = field.find("flow")
            trajectory = field.find("trajectory")
            line_integral = field.find("line_integral")
            euclidean_gradient = field.find("euclidean_gradient")
            zero_curl = field.find("zero_curl")
            entropy = field.find("entropy")
            entropy_gradient = field.find("entropy_gradient")
            free_energy = field.find("free_energy")
            fokker_planck = field.find("fokker_planck")
            inscribed_mesh = field.find("inscribed_mesh")
            boundary_reset = field.find("boundary_reset")
            tensor_field = field.find("tensor_field")
            tensor_transform = field.find("tensor_transform")
            metric_tensor = field.find("metric_tensor")
            jacobian = field.find("jacobian")
        else:
            hamiltonian = plane_wave = flow = trajectory = line_integral = euclidean_gradient = zero_curl = entropy = entropy_gradient = free_energy = fokker_planck = inscribed_mesh = boundary_reset = tensor_field = tensor_transform = metric_tensor = jacobian = None
        cases.append(("continuous_field_math_sources_declared",
                      hamiltonian is not None and
                      hamiltonian.attrib.get("symbol") == "GraphHamiltonian" and
                      hamiltonian.attrib.get("mathml") == "data/grammar/vectors.mml" and
                      plane_wave is not None and
                      plane_wave.attrib.get("symbol") == "PlaneWaveSuperposition" and
                      flow is not None and flow.attrib.get("symbol") == "GradientFlow" and
                      trajectory is not None and trajectory.attrib.get("symbol") == "DifferentiableCurve" and
                      line_integral is not None and line_integral.attrib.get("symbol") == "SemanticDisplacement" and
                      euclidean_gradient is not None and euclidean_gradient.attrib.get("symbol") == "EuclideanGradient" and
                      zero_curl is not None and zero_curl.attrib.get("symbol") == "ZeroCurlConstraint" and
                      entropy is not None and entropy.attrib.get("symbol") == "ShannonEntropy" and
                      entropy_gradient is not None and entropy_gradient.attrib.get("symbol") == "EntropyGradient" and
                      free_energy is not None and free_energy.attrib.get("symbol") == "FreeEnergyFlow" and
                      fokker_planck is not None and fokker_planck.attrib.get("symbol") == "FokkerPlanckFlow" and
                      inscribed_mesh is not None and
                      inscribed_mesh.attrib.get("symbol") == "EulerTriangleInequality" and
                      inscribed_mesh.attrib.get("schema") == "data/schema/entropic-well.schema.json" and
                      boundary_reset is not None and
                      boundary_reset.attrib.get("symbol") == "StructuralBoundaryReset"))
        cases.append(("continuous_field_tensor_bundle_declared",
                      tensor_field is not None and
                      tensor_field.attrib.get("symbol") == "TensorField" and
                      "tensor_rank_p" in tensor_field.attrib.get("gml_attrs", "") and
                      tensor_transform is not None and
                      tensor_transform.attrib.get("symbol") == "TensorTransform" and
                      tensor_transform.attrib.get("mathml") == "data/grammar/vectors.mml" and
                      metric_tensor is not None and
                      metric_tensor.attrib.get("symbol") == "MetricTensor" and
                      jacobian is not None and
                      jacobian.attrib.get("symbol") == "JacobianMatrix"))
        cases.append(("continuous_field_authority_boundary_declared",
                      "Continuous fields constrain provider geometry" in
                      field_laws.get("field-not-truth", "") and
                      "cannot promote relations or mutate B" in
                      field_laws.get("curve-not-truth", "") and
                      "do not verify facts" in
                      field_laws.get("entropy-not-truth", "") and
                      "cannot authorize relations" in
                      field_laws.get("inscribed-mesh-not-truth", "") and
                      "cannot manufacture semantic truth" in
                      field_laws.get("tensor-not-truth", "")))

        training_gradient = root.find(".//training_gradient_contract")
        training_laws = {
            law.attrib.get("id"): (law.text or "")
            for law in root.findall(".//training_gradient_contract/law")
        }
        if training_gradient is not None:
            tg_params = training_gradient.find("parameters")
            tg_loss = training_gradient.find("loss")
            tg_grad = training_gradient.find("gradient")
            tg_update = training_gradient.find("update")
            tg_clip = training_gradient.find("clip")
        else:
            tg_params = tg_loss = tg_grad = tg_update = tg_clip = None
        cases.append(("training_gradient_contract_declared",
                      training_gradient is not None and
                      training_gradient.attrib.get("id") == "vector:training_gradient:v1" and
                      training_gradient.attrib.get("semantic_authority") == "false" and
                      training_gradient.attrib.get("projection_only") == "true" and
                      tg_params is not None and tg_params.attrib.get("symbol") == "ProviderParameters" and
                      tg_loss is not None and tg_loss.attrib.get("symbol") == "ProjectionLoss" and
                      tg_grad is not None and tg_grad.attrib.get("symbol") == "TrainingGradient" and
                      tg_update is not None and tg_update.attrib.get("symbol") == "ParameterUpdate" and
                      tg_clip is not None and tg_clip.attrib.get("symbol") == "GradientClip" and
                      "not FoldDelta" in
                      training_laws.get("gradient-not-fold-delta", "")))

        container = root.find(".//container[@key='vector:projection_container:v1']")
        cases.append(("vector_container_declared",
                      container is not None and
                      container.attrib.get("semantic_authority") == "false" and
                      container.attrib.get("topology_source") == "gml:topology_contract"))
        tensor_container = root.find(".//container[@key='tensor:projection_container:v1']")
        cases.append(("tensor_container_declared",
                      tensor_container is not None and
                      tensor_container.attrib.get("owner") == "TENSOR-µ" and
                      tensor_container.attrib.get("semantic_authority") == "false" and
                      tensor_container.attrib.get("topology_source") == "gml:topology_contract" and
                      tensor_container.attrib.get("legacy_container") == "vector:projection_container:v1"))

        external = [p for p in providers.values()
                    if p.attrib.get("source", "").startswith("E:/")]
        cases.append(("external_sources_optional",
                      bool(external) and all(p.attrib.get("require") == "false" for p in external)))

        return verdict("vectors_xml", cases)
    except Exception as exc:
        return verdict("vectors_xml", cases, str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
