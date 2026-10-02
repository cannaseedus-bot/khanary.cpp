"""Validate the METRIC-µ graph, manifest, and MathML contracts.

Suite name: metric_graph  (must match run_conformance.py required set)
"""

from pathlib import Path
import json
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).parent))
from conformance_common import verdict  # noqa: E402
from check_gml_validator import validate_gml  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
MML_NS = {"mml": "http://www.w3.org/1998/Math/MathML"}


def main():
    cases = []
    try:
        registry = ET.parse(ROOT / "metric_graph.xml").getroot()
        cases.append(("metric_registry_root",
                      registry.tag == "metric_registry" and
                      registry.attrib.get("id") == "khanary-metric-graph" and
                      registry.attrib.get("semantic_authority") == "false" and
                      registry.attrib.get("topology_authority") == "false"))

        invariants = {elem.attrib.get("id") for elem in registry.findall(".//invariant")}
        cases.append(("metric_registry_declares_track_locality",
                      {"track_compartment_first", "no_global_metric_space",
                       "cross_track_bridge_required", "chen_remains_verifier"}.issubset(invariants)))

        track_contract = registry.find(".//track_compartment_contract")
        metric_contract = registry.find(".//metric_contract")
        cases.append(("track_compartment_contract_declared",
                      track_contract is not None and
                      track_contract.attrib.get("symbol") == "Track_tau" and
                      track_contract.find("track") is not None and
                      track_contract.find("local_metric") is not None and
                      track_contract.find("bridge") is not None))
        cases.append(("metric_contract_is_per_track",
                      metric_contract is not None and
                      metric_contract.attrib.get("symbol") == "M_tau" and
                      metric_contract.attrib.get("over") == "Track_tau" and
                      metric_contract.find("cross_track_relevance") is not None))
        training_objective = registry.find(".//training_objective")
        cases.append(("metric_training_objective_declared",
                      training_objective is not None and
                      training_objective.attrib.get("id") == "metric:contrastive_risk:v1" and
                      training_objective.attrib.get("semantic_authority") == "false" and
                      training_objective.attrib.get("topology_authority") == "false" and
                      training_objective.attrib.get("verdict_source") == "Ch'en" and
                      training_objective.find("loss") is not None and
                      training_objective.find("risk_penalty") is not None and
                      training_objective.find("differential_loss") is not None))

        node_classes = {elem.attrib.get("id") for elem in registry.findall(".//heterogeneous_graph_contract/node_class")}
        cases.append(("heterogeneous_node_classes_declared",
                      {"V_gram", "V_track", "V_fold", "V_mu", "V_mesh"}.issubset(node_classes)))
        mesh_rule = "\n".join((elem.text or "") for elem in registry.findall(".//heterogeneous_graph_contract/edge_rule"))
        cases.append(("mesh_discovery_not_execution_declared",
                      "permits discovery" in mesh_rule and "not execution" in mesh_rule))

        providers = {elem.attrib.get("id"): elem for elem in registry.findall(".//provider")}
        cases.append(("metric_provider_set_declared",
                      {"metric-graph-provider", "vector-provider", "sdr-provider",
                       "graph-provider", "scx-provider"}.issubset(providers)))
        cases.append(("vector_is_provider_not_root",
                      providers.get("vector-provider") is not None and
                      providers["vector-provider"].attrib.get("owner") == "VECTOR-µ" and
                      providers["vector-provider"].attrib.get("semantic_authority") == "false"))

        manifest = json.loads((ROOT / "data/manifest/metric.manifest.json").read_text(encoding="utf-8"))
        cases.append(("metric_manifest_declares_core_keys",
                      {"METRIC:", "@metric", "METRIC-µ", "metric:contract",
                       "metric:track-compartment", "metric:cross-track-bridge",
                       "metric:heterogeneous-graph", "metric:graph:v1",
                       "metric:contrastive-loss", "metric:bridge-loss",
                       "metric:differential-risk-loss"}.issubset(manifest)))
        cases.append(("metric_manifest_declares_no_global_union_metric",
                      "no_global_union_metric" in manifest.get("metric:track-compartment", []) and
                      "mesh_connectivity_permits_discovery_not_execution" in manifest.get("metric:heterogeneous-graph", [])))
        cases.append(("metric_manifest_declares_contrastive_risk_loss",
                      "no_global_all_grams_negative_pool" in manifest.get("metric:contrastive-loss", []) and
                      "Bridge_alpha_beta_required" in manifest.get("metric:bridge-loss", []) and
                      "L_risk=max(0,Risk(theta,delta)-risk_bound)^2" in manifest.get("metric:differential-risk-loss", []) and
                      "DeltaMetric!=DeltaTopology" in manifest.get("metric:differential-risk-loss", [])))

        mml = ET.parse(ROOT / "data/grammar/metric_graph.mml").getroot()
        annotations = "\n".join((elem.text or "") for elem in mml.findall(".//mml:annotation", MML_NS))
        csymbols = {(elem.attrib.get("cd"), elem.text or "") for elem in mml.findall(".//mml:csymbol", MML_NS)}
        cases.append(("metric_mml_declares_track_metric",
                      mml.tag == "{http://www.w3.org/1998/Math/MathML}math" and
                      ("urn:kuhul:metric-graph:v1", "TrackMetric") in csymbols and
                      "M_tau = METRIC_mu(Track_tau)" in annotations and
                      "no required global semantic metric" in annotations))
        cases.append(("metric_mml_declares_relevance_switch",
                      ("urn:kuhul:metric-graph:v1", "TrackRelevanceSwitch") in csymbols and
                      "R = R_tau(Q,K) inside one admitted track" in annotations and
                      "R = 0/NA when no track bridge exists" in annotations))
        cases.append(("metric_mml_declares_delta_metric_not_topology",
                      ("urn:kuhul:metric-graph:v1", "DeltaMetric") in csymbols and
                      ("urn:kuhul:metric-graph:v1", "DeltaTopology") in csymbols and
                      "Only Proposal -> Ch'en -> XCFE -> Xul may commit topology B" in annotations))
        cases.append(("metric_mml_declares_contrastive_objective",
                      ("urn:kuhul:metric-graph:v1", "MetricScore") in csymbols and
                      ("urn:kuhul:metric-graph:v1", "MetricContrastiveLoss") in csymbols and
                      "L_contrast = -log" in annotations and
                      "no global all-grams negative pool" in annotations))
        cases.append(("metric_mml_declares_bridge_diff_risk_objective",
                      ("urn:kuhul:metric-graph:v1", "BridgeTraversalCost") in csymbols and
                      ("urn:kuhul:metric-graph:v1", "DifferentialDriftLoss") in csymbols and
                      ("urn:kuhul:metric-graph:v1", "RiskPenalty") in csymbols and
                      ("urn:kuhul:metric-graph:v1", "MetricTotalLoss") in csymbols and
                      "L_diff = ||Projection(DeltaG) - DeltaF||^2" in annotations and
                      "L_total = alpha*L_contrast + beta*L_bridge + gamma*L_diff + eta*L_risk" in annotations))

        gml_text = (ROOT / "data/metric_graph.gml").read_text(encoding="utf-8")
        gml_result = validate_gml(gml_text, "metric_graph")
        cases.append(("metric_gml_validates",
                      gml_result["grade"] == "VALID" and not gml_result["errors"]))
        cases.append(("metric_gml_declares_track_nodes",
                      all(label in gml_text for label in ("Track_MATH", "Track_ELIZA", "Track_CODE", "Track_JOKE"))))
        cases.append(("metric_gml_declares_heterogeneous_nodes",
                      all(label in gml_text for label in ("GramNode:why@JOKE", "GramNode:why@ADVISOR",
                                                          "FoldNode:Yax", "MicronautNode:JOKE-µ",
                                                          "MeshNode:delivery_reflection"))))
        cases.append(("metric_gml_declares_mesh_discovery_edges",
                      "mesh_discovers_eliza_reflection" in gml_text and
                      "authorized_cross_track_metric" in gml_text and
                      "joke_why_to_delivery_mesh" in gml_text))

        return verdict("metric_graph", cases)
    except Exception as exc:
        cases.append(("metric_graph_exception", False))
        return verdict("metric_graph", cases, str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
