"""GML semantic validator conformance suite.

Validates the rules declared in data/grammar/gml.validator.json against
fixture vectors (data/conformance/gml_validator.json) and all shipped GML
files (data/*.gml).

Suite name: gml_validator  (must match run_conformance.py required set)
"""
import hashlib
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from conformance_common import verdict

ROOT = Path(__file__).resolve().parents[1]

# Canonical verbs from src/micronaut/relation_isa.h (GML subset)
CANONICAL_VERBS = frozenset([
    "@related_to", "@delegates_to", "@requires", "@part_of", "@defines",
    "@is_a", "@causes", "@contradicts", "@has_part", "@has_property",
    "@possesses", "@value", "@imports", "@calls", "@exposes",
])

# ── Minimal GML tokenizer ─────────────────────────────────────────────────────

_TOKEN_RE = re.compile(
    r'"(?:[^"\\]|\\.)*"'                      # quoted string
    r'|[-+]?\d+\.\d*(?:[eE][-+]?\d+)?'       # float with decimal point
    r'|[-+]?\d+[eE][-+]?\d+'                  # float with exponent only
    r'|[-+]?\d+'                               # integer
    r'|[A-Za-z_][A-Za-z0-9_]*'               # identifier
    r'|[\[\]]',                                # brackets
    re.ASCII
)


def _tokenize(s):
    return _TOKEN_RE.findall(s)


def _parse_value(tok):
    if tok.startswith('"'):
        return tok[1:-1].replace('\\"', '"')
    try:
        return int(tok)
    except ValueError:
        pass
    try:
        return float(tok)
    except ValueError:
        pass
    return tok


def _parse_block(toks, i):
    """Return (pairs_list, i_after_closing_bracket).
    i points to the first token inside the block (after '[')."""
    pairs = []
    while i < len(toks) and toks[i] != ']':
        key = toks[i]
        i += 1
        if i >= len(toks):
            break
        if toks[i] == '[':
            i += 1
            sub, i = _parse_block(toks, i)
            if i < len(toks) and toks[i] == ']':
                i += 1
            val = sub  # list of (key, value) pairs → converted below
        else:
            val = _parse_value(toks[i])
            i += 1
        pairs.append((key, val))
    return pairs, i


def _pairs_to_dict(pairs):
    d = {}
    for k, v in pairs:
        if isinstance(v, list):
            v = _pairs_to_dict(v)
        if k in d:
            existing = d[k]
            if not isinstance(existing, list):
                existing = d[k] = [existing]
            existing.append(v)
        else:
            d[k] = v
    return d


def _parse_gml(text):
    toks = _tokenize(text)
    if not toks or toks[0] != 'graph':
        raise ValueError("expected 'graph'")
    if len(toks) < 2 or toks[1] != '[':
        raise ValueError("expected '[' after graph")
    pairs, _ = _parse_block(toks, 2)
    g = _pairs_to_dict(pairs)
    nodes = g.pop('node', [])
    if isinstance(nodes, dict):
        nodes = [nodes]
    edges = g.pop('edge', [])
    if isinstance(edges, dict):
        edges = [edges]
    return {'attrs': g, 'nodes': nodes, 'edges': edges}


# ── Stem derivation ───────────────────────────────────────────────────────────

def _derive_stem(filename):
    raw = re.sub(r'[^A-Za-z0-9]', '_', Path(filename).stem.lower())
    return raw


# ── Edit distance for near-miss detection ────────────────────────────────────

def _edit_dist(a, b):
    m, n = len(a), len(b)
    row = list(range(n + 1))
    for i in range(1, m + 1):
        prev, row[0] = row[0], i
        for j in range(1, n + 1):
            prev, row[j] = row[j], min(
                row[j] + 1, row[j - 1] + 1,
                prev + (0 if a[i - 1] == b[j - 1] else 1)
            )
    return row[n]


def _is_near_miss(verb):
    return any(_edit_dist(verb, cv) <= 2 for cv in CANONICAL_VERBS)


def _unwrap_cdata_payload(text):
    start = text.find('<![CDATA[')
    if start < 0:
        return text
    start += len('<![CDATA[')
    end = text.find(']]>', start)
    if end < 0:
        return None
    return text[start:end]


def _cso_id_for_source(text):
    return 'cso://' + hashlib.sha256(text.encode('utf-8')).hexdigest()


PROFILE_FIELD_KEYS = frozenset([
    'brain_profile', 'personality', 'capability_grams', 'progression',
])
PROFILE_AUTHORITY_KEYS = frozenset([
    'semantic_authority', 'topology_authority', 'commit_authority',
    'verification_authority', 'may_verify_truth', 'may_commit_topology',
    'can_verify', 'can_commit',
])
DEDUCTIVE_CLAIM_KEYS = frozenset([
    'logical_consequence', 'semantic_entails', 'models_phi', 'sigma_models_phi',
])
DEDUCTIVE_PROOF_KEYS = frozenset([
    'deductive_proof', 'proof_trace', 'phase_proof_trace', 'sigma_proves_phi',
])
AXIOM_EXTENSION_KEYS = frozenset([
    'axiom_extension', 'nonlogical_axiom', 'sigma_extension',
])
LAW_CONTRADICTION_KEYS = frozenset([
    'contradicts_law', 'law_contradiction', 'breaks_primary_law',
])
MISS_STATE_KEYS = frozenset([
    'epistemic_state', 'observation_state', 'chen_state', 'resolution_state',
    'miss_kind',
])
CONTRADICTION_STATE_KEYS = frozenset([
    'truth_verdict', 'evidence_verdict', 'epistemic_state', 'chen_state',
    'resolution_state',
])
ALGEBRAIC_RANK_KEY_PAIRS = (
    ('matrix_rank_substrate', 'matrix_rank_transformed'),
    ('substrate_matrix_rank', 'transformed_matrix_rank'),
    ('signature_rank_substrate', 'signature_rank_transformed'),
)
ALGEBRAIC_ISOMORPHISM_KEYS = frozenset([
    'signature_isomorphism_preserved', 'signature_isomorphic',
    'preserves_signature', 'functor_preserves_signature',
])


def _truthy(value):
    return value is True or value in (1, 1.0, '1', 'true', 'True')


def _values_for(item, keys):
    values = []
    for key in keys:
        if key not in item:
            continue
        value = item.get(key)
        if isinstance(value, list):
            values.extend(value)
        else:
            values.append(value)
    return values


def _claim_present(value):
    if isinstance(value, list):
        return any(_claim_present(v) for v in value)
    if isinstance(value, dict):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() not in ('', '0', 'false', 'no', 'none', 'null')
    return _truthy(value)


def _proof_present(item):
    return any(_claim_present(value) for value in _values_for(item, DEDUCTIVE_PROOF_KEYS))


def _upper_token(value):
    if isinstance(value, list):
        return [_upper_token(v) for v in value]
    if value is None:
        return ''
    return str(value).replace('-', '_').replace(' ', '_').upper()


def _contains_state(values, predicate):
    for value in values:
        token = _upper_token(value)
        if isinstance(token, list):
            if _contains_state(token, predicate):
                return True
        elif predicate(token):
            return True
    return False


def _profile_metric_out_of_range(item):
    grams = item.get('capability_grams')
    if isinstance(grams, dict):
        for value in grams.values():
            if not _is_number(value) or not (0.0 <= float(value) <= 1.0):
                return True
    for key, value in item.items():
        if (key.startswith('capability_gram_') or key == 'gram_score') and \
                (not _is_number(value) or not (0.0 <= float(value) <= 1.0)):
            return True
    progression = item.get('progression')
    level = progression.get('level') if isinstance(progression, dict) else item.get('progression_level')
    if level is not None and (not _is_number(level) or float(level) < 0.0):
        return True
    return False


def _matrix_rank(value):
    if not _is_number(value):
        return None
    numeric = float(value)
    if numeric < 0.0 or abs(numeric - round(numeric)) > _EPS:
        return None
    return int(round(numeric))


def _validate_algebraic_interpretation(item, errors):
    # Explicit non-isomorphism is fatal: the fold algebra would not be a lawful
    # interpretation of the same semantic signature.
    for key in ALGEBRAIC_ISOMORPHISM_KEYS:
        if key in item and not _truthy(item.get(key)):
            _append_error(errors, 'SIGNATURE_FUNCTOR_FRACTION')
            break

    # Optional rank diagnostics are checked only when both sides are present.
    # This keeps ordinary lenient GML valid while making algebraic overlays prove
    # that they preserve the substrate signature rank.
    for substrate_key, transformed_key in ALGEBRAIC_RANK_KEY_PAIRS:
        if substrate_key not in item and transformed_key not in item:
            continue
        if substrate_key not in item or transformed_key not in item:
            continue
        substrate_rank = _matrix_rank(item.get(substrate_key))
        transformed_rank = _matrix_rank(item.get(transformed_key))
        if substrate_rank is None or transformed_rank is None or substrate_rank != transformed_rank:
            _append_error(errors, 'SIGNATURE_FUNCTOR_FRACTION')


_GAMMA_KEY_RE = re.compile(r'^gamma_(\d+)_(\d+)_(\d+)$')


def _append_error(errors, code):
    if code not in errors:
        errors.append(code)


def _ricci_tolerance(item):
    value = item.get('ricci_tolerance')
    if value is None:
        return 1.0e-4
    if not _is_number(value) or float(value) < 0.0:
        return 0.0
    return float(value)


def _validate_tensor_connection(item, errors):
    tolerance = _ricci_tolerance(item)

    gamma = item.get('christoffel')
    if gamma is not None:
        if not isinstance(gamma, dict):
            _append_error(errors, 'CHRISTOFFEL_NOT_NUMERIC')
        else:
            values = {}
            for key, value in gamma.items():
                match = _GAMMA_KEY_RE.match(str(key))
                if not match or not _is_number(value):
                    _append_error(errors, 'CHRISTOFFEL_NOT_NUMERIC')
                    continue
                values[tuple(int(part) for part in match.groups())] = float(value)
            for (k, i, j), value in values.items():
                mirror = values.get((k, j, i))
                if mirror is not None and abs(value - mirror) > tolerance:
                    _append_error(errors, 'CHRISTOFFEL_TORSION_NONZERO')
                    break

    covariant = item.get('covariant_derivative')
    if covariant is not None:
        if not isinstance(covariant, dict):
            _append_error(errors, 'COVARIANT_RANK_DRIFT')
        else:
            in_p = covariant.get('input_rank_p')
            in_q = covariant.get('input_rank_q')
            out_p = covariant.get('output_rank_p')
            out_q = covariant.get('output_rank_q')
            if None not in (in_p, in_q, out_p, out_q):
                if not all(isinstance(v, int) and v >= 0 for v in (in_p, in_q, out_p, out_q)):
                    _append_error(errors, 'COVARIANT_RANK_DRIFT')
                elif out_p != in_p or out_q != in_q + 1:
                    _append_error(errors, 'COVARIANT_RANK_DRIFT')
            residual = covariant.get('residual')
            if residual is not None:
                if not _is_number(residual) or abs(float(residual)) > tolerance:
                    _append_error(errors, 'RICCI_IDENTITY_VIOLATION')

    residual = item.get('ricci_identity_residual')
    if residual is not None:
        if not _is_number(residual) or abs(float(residual)) > tolerance:
            _append_error(errors, 'RICCI_IDENTITY_VIOLATION')


# ── Validator ─────────────────────────────────────────────────────────────────

def _check_range(value, lo, hi):
    return isinstance(value, (int, float)) and lo <= float(value) <= hi

MUTATION_OPS = frozenset(["insert", "delete", "modify", "unchanged"])
VERTEX_BC = frozenset(["kirchhoff", "dirichlet", "robin"])
SINGULARITY_TYPES = frozenset(["source", "sink", "saddle", "ordinary"])
TEXT_BOUNDARIES = frozenset(["sentence", "paragraph", "article", "section", "none"])
FIELD_KINDS = frozenset([
    "flow", "trajectory", "euclidean_gradient", "entropy_gradient",
    "free_energy", "riemannian_flow", "plane_wave", "quantum",
    "tensor_field", "hyperbolic_event"
])
SPACE_SIGNATURES = frozenset([
    "Minkowski_Minus_Plus_Plus_Plus",
    "Minkowski_Plus_Minus_Minus_Minus",
])
INTERVAL_CLASSES = frozenset([
    "timelike_candidate",
    "spacelike_candidate",
    "lightlike_boundary",
    "unclassified",
])
HYPERBOLIC_EVENT_KEYS = frozenset([
    "space_signature", "event_role", "rapidity_rho", "minkowski_interval",
    "interval_class", "minkowski_t", "minkowski_x", "minkowski_y",
    "minkowski_z", "direction_x", "direction_y", "direction_z",
    "hyperbolic_distance",
])
_EPS = 1.0e-9


def _is_number(value):
    return isinstance(value, (int, float)) and (
        not isinstance(value, float) or math.isfinite(value)
    )


def _interval_class_for(interval):
    if interval < -_EPS:
        return "timelike_candidate"
    if interval > _EPS:
        return "spacelike_candidate"
    return "lightlike_boundary"


def _minkowski_interval_from_coords(item, signature):
    values = [item.get(key) for key in ("minkowski_t", "minkowski_x", "minkowski_y", "minkowski_z")]
    present = [value is not None for value in values]
    if not any(present):
        return None, False
    if not all(present):
        return None, True
    if not all(_is_number(value) for value in values):
        return None, False
    t, x, y, z = (float(value) for value in values)
    if signature == "Minkowski_Plus_Minus_Minus_Minus":
        return (t * t) - (x * x) - (y * y) - (z * z), False
    return -(t * t) + (x * x) + (y * y) + (z * z), False


def _validate_hyperbolic_event_item(item, errors, warnings):
    field_kind = item.get("field_kind")
    has_hyperbolic_fields = field_kind == "hyperbolic_event" or any(
        key in item for key in HYPERBOLIC_EVENT_KEYS
    )
    if not has_hyperbolic_fields:
        return

    signature = item.get("space_signature")
    if signature is not None and signature not in SPACE_SIGNATURES:
        _append_error(errors, "SPACE_SIGNATURE_INVALID")
    if signature is None:
        signature = "Minkowski_Minus_Plus_Plus_Plus"

    rapidity = item.get("rapidity_rho")
    if rapidity is not None and (not _is_number(rapidity) or float(rapidity) < 0.0):
        _append_error(errors, "RAPIDITY_RHO_NEGATIVE")

    interval = item.get("minkowski_interval")
    if interval is not None and not _is_number(interval):
        _append_error(errors, "MINKOWSKI_INTERVAL_NOT_NUMERIC")

    interval_class = item.get("interval_class")
    if interval_class is not None and interval_class not in INTERVAL_CLASSES:
        _append_error(errors, "INVALID_INTERVAL_CLASS")
    if _is_number(interval) and interval_class not in (None, "unclassified"):
        expected = _interval_class_for(float(interval))
        if interval_class != expected:
            _append_error(errors, "MINKOWSKI_INTERVAL_CLASS_MISMATCH")

    coord_interval, partial_coords = _minkowski_interval_from_coords(item, signature)
    if partial_coords:
        if "PARTIAL_MINKOWSKI_EVENT_VECTOR" not in warnings:
            warnings.append("PARTIAL_MINKOWSKI_EVENT_VECTOR")
    coord_values = [item.get(key) for key in ("minkowski_t", "minkowski_x", "minkowski_y", "minkowski_z")]
    if all(value is not None for value in coord_values) and coord_interval is None:
        _append_error(errors, "MINKOWSKI_EVENT_VECTOR_NOT_NUMERIC")
    if coord_interval is not None and _is_number(interval):
        if abs(coord_interval - float(interval)) > 1.0e-6:
            _append_error(errors, "MINKOWSKI_INTERVAL_MISMATCH")

    direction_values = [item.get(key) for key in ("direction_x", "direction_y", "direction_z")]
    present_direction = [value is not None for value in direction_values]
    if any(present_direction) and not all(present_direction):
        if "PARTIAL_HYPERBOLIC_DIRECTION" not in warnings:
            warnings.append("PARTIAL_HYPERBOLIC_DIRECTION")
    if all(present_direction):
        if not all(_is_number(value) for value in direction_values):
            _append_error(errors, "HYPERBOLIC_DIRECTION_NOT_NUMERIC")
        elif sum(float(value) * float(value) for value in direction_values) <= _EPS:
            _append_error(errors, "HYPERBOLIC_DIRECTION_ZERO")

    distance = item.get("hyperbolic_distance")
    if distance is not None and (not _is_number(distance) or float(distance) < 0.0):
        _append_error(errors, "HYPERBOLIC_DISTANCE_NEGATIVE")

    if any(_truthy(item.get(key)) for key in (
        "semantic_authority", "topology_authority", "commit_authority",
        "verification_authority", "may_verify_truth", "may_commit_topology",
        "can_verify", "can_commit",
    )):
        _append_error(errors, "HYPERBOLIC_EVENT_AUTHORITY_FORBIDDEN")


_LORENTZ_COMPONENT_KEYS = ("c0", "c1", "c2", "c3")
_LORENTZ_IDENTITY = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


def _lorentz_enabled(value):
    if value is None:
        return False, True
    if isinstance(value, bool):
        return value, True
    if value in (0, 1, 0.0, 1.0, "0", "1", "false", "true", "False", "True"):
        return value in (1, 1.0, "1", "true", "True"), True
    return False, False


def _lorentz_row_values(row):
    if not isinstance(row, dict):
        return None
    values = []
    for key in _LORENTZ_COMPONENT_KEYS:
        value = row.get(key)
        if not _is_number(value):
            return None
        values.append(float(value))
    return tuple(values)


def _validate_lorentz_boost(item, errors):
    boost = item.get("lorentz_boost")
    if boost is None:
        return
    if not isinstance(boost, dict):
        _append_error(errors, "LORENTZ_ROW_NOT_NUMERIC")
        return

    enabled, enabled_ok = _lorentz_enabled(boost.get("enabled", boost.get("lorentz_enabled")))
    if not enabled_ok:
        _append_error(errors, "LORENTZ_ENABLED_NOT_BOOL")

    alignment = boost.get("alignment_mix", boost.get("boost_alignment_mix"))
    if alignment is not None and (not _is_number(alignment) or not (0.0 <= float(alignment) <= 1.0)):
        _append_error(errors, "LORENTZ_ALIGNMENT_OUT_OF_RANGE")

    mass_scale = boost.get("mass_scale", boost.get("boost_mass_scale"))
    if mass_scale is not None and (not _is_number(mass_scale) or float(mass_scale) < 0.0):
        _append_error(errors, "LORENTZ_MASS_SCALE_NEGATIVE")

    rows = []
    for index, key in enumerate(("row0", "row1", "row2", "row3")):
        row = boost.get(key)
        if row is None:
            rows.append(None)
            continue
        values = _lorentz_row_values(row)
        if values is None:
            _append_error(errors, "LORENTZ_ROW_NOT_NUMERIC")
        rows.append(values)

    if enabled and any(row is None for row in rows):
        _append_error(errors, "LORENTZ_ROW_INCOMPLETE")

    if not enabled:
        tolerance = item.get("lorentz_tolerance", item.get("ricci_tolerance", 1.0e-4))
        if not _is_number(tolerance) or float(tolerance) < 0.0:
            tolerance = 0.0
        for index, row in enumerate(rows):
            if row is None:
                continue
            expected = _LORENTZ_IDENTITY[index]
            if any(abs(row[i] - expected[i]) > float(tolerance) for i in range(4)):
                _append_error(errors, "LORENTZ_NOT_IDENTITY_WHEN_DISABLED")
                break


def validate_gml(gml_text, stem='__default__'):
    """Return {'grade': VALID|PARSED, 'errors': [...], 'warnings': [...]}."""
    errors = []
    warnings = []

    try:
        graph = _parse_gml(gml_text)
    except Exception:
        return {'grade': 'PARSED', 'errors': ['PARSE_ERROR'], 'warnings': []}

    nodes = graph['nodes']
    edges = graph['edges']
    node_by_id = {}

    # ── V1: node id required and unique ──────────────────────────────────────
    seen_ids = set()
    for node in nodes:
        nid = node.get('id')
        if nid is None:
            if 'MISSING_NODE_ID' not in errors:
                errors.append('MISSING_NODE_ID')
        elif nid in seen_ids:
            if 'DUPLICATE_NODE_ID' not in errors:
                errors.append('DUPLICATE_NODE_ID')
        else:
            seen_ids.add(nid)
            node_by_id[nid] = node

    # ── H1: inline GPU kernel CDATA must match its CSO identity ─────────────
    for node in nodes:
        for source_key in ('kernel_source', 'shader_source'):
            source = node.get(source_key)
            if not isinstance(source, str) or '<![CDATA[' not in source:
                continue
            unwrapped = _unwrap_cdata_payload(source)
            declared = node.get('cso_id', node.get('cso', ''))
            if not isinstance(declared, str):
                declared = ''
            if unwrapped is None or not unwrapped or not declared or \
                    declared != _cso_id_for_source(unwrapped):
                if 'CSO_INTEGRITY_MISMATCH' not in errors:
                    errors.append('CSO_INTEGRITY_MISMATCH')

    # ── P1/P2: profile/progression metrics are projections, not authority ────
    for item in list(nodes) + list(edges):
        has_profile = any(key in item for key in PROFILE_FIELD_KEYS)
        if has_profile and any(_truthy(item.get(key)) for key in PROFILE_AUTHORITY_KEYS):
            if 'PROFILE_AUTHORITY_FORBIDDEN' not in errors:
                errors.append('PROFILE_AUTHORITY_FORBIDDEN')
        if _profile_metric_out_of_range(item):
            if 'PROFILE_METRIC_OUT_OF_RANGE' not in errors:
                errors.append('PROFILE_METRIC_OUT_OF_RANGE')

    # ── L1-L3: deductive-system boundary checks ─────────────────────────────
    # Logical consequence claims need an executable/projection proof trace.
    # MISS remains an observation deficit; it must not be collapsed into
    # CONTRADICTED without negative evidence. Axiom extensions may add domain
    # facts only when they do not break the primary law layer.
    for item in [graph['attrs']] + list(nodes) + list(edges):
        has_consequence_claim = any(
            _claim_present(value)
            for value in _values_for(item, DEDUCTIVE_CLAIM_KEYS)
        )
        if has_consequence_claim and not _proof_present(item):
            if 'DEDUCTIVE_PROOF_REQUIRED' not in errors:
                errors.append('DEDUCTIVE_PROOF_REQUIRED')

        has_axiom_extension = any(
            _claim_present(value)
            for value in _values_for(item, AXIOM_EXTENSION_KEYS)
        )
        has_law_contradiction = any(
            _claim_present(value)
            for value in _values_for(item, LAW_CONTRADICTION_KEYS)
        )
        if has_axiom_extension and has_law_contradiction:
            if 'AXIOM_EXTENSION_CONTRADICTS_LAW' not in errors:
                errors.append('AXIOM_EXTENSION_CONTRADICTS_LAW')

        miss_values = _values_for(item, MISS_STATE_KEYS)
        contradiction_values = _values_for(item, CONTRADICTION_STATE_KEYS)
        has_miss = _contains_state(miss_values, lambda token: token.startswith('MISS'))
        has_contradiction = _contains_state(
            contradiction_values,
            lambda token: token in ('CONTRADICTED', 'CONTRADICTION', 'NEGATIVE_EVIDENCE')
                          or token.startswith('CONTRADICTED')
        )
        if has_miss and has_contradiction:
            if 'MISS_COLLAPSED_TO_CONTRADICTION' not in errors:
                errors.append('MISS_COLLAPSED_TO_CONTRADICTION')

    # ── A1: interpretation algebras preserve signature rank ─────────────────
    for item in [graph['attrs']] + list(nodes) + list(edges):
        _validate_algebraic_interpretation(item, errors)

    # ── HE1-HE7: hyperbolic key-grammar event projection checks ─────────────
    for item in [graph['attrs']] + list(nodes) + list(edges):
        _validate_hyperbolic_event_item(item, errors, warnings)

    # ── V2/V3: edge endpoints required and resolvable ─────────────────────────
    for edge in edges:
        src = edge.get('source')
        tgt = edge.get('target')
        if src is None or tgt is None:
            if 'MISSING_EDGE_ENDPOINT' not in errors:
                errors.append('MISSING_EDGE_ENDPOINT')
        else:
            if src not in seen_ids or tgt not in seen_ids:
                if 'INVALID_EDGE_ENDPOINT' not in errors:
                    errors.append('INVALID_EDGE_ENDPOINT')

    # ── V5: file-stem ident safety ────────────────────────────────────────────
    if stem and stem[0].isdigit():
        if 'STEM_STARTS_WITH_DIGIT' not in errors:
            errors.append('STEM_STARTS_WITH_DIGIT')

    # ── R1–R5: relation authority ─────────────────────────────────────────────
    for edge in edges:
        verb = edge.get('verb')
        relation = edge.get('relation')

        if verb is not None and relation is not None:
            if 'BOTH_VERB_AND_RELATION_PRESENT' not in warnings:
                warnings.append('BOTH_VERB_AND_RELATION_PRESENT')

        # effective verb: verb wins; relation is fallback candidate
        eff = verb if verb is not None else relation
        if verb is None and relation is not None:
            if 'RELATION_ALIAS_USED' not in warnings:
                warnings.append('RELATION_ALIAS_USED')

        if eff is not None and eff not in CANONICAL_VERBS:
            if 'INFER_CANDIDATE_USED' not in warnings:
                warnings.append('INFER_CANDIDATE_USED')
            if _is_near_miss(eff):
                if 'NEAR_MISS_VERB' not in warnings:
                    warnings.append('NEAR_MISS_VERB')

    # ── W1–W4: WCR unit-float ranges ─────────────────────────────────────────
    for node in nodes:
        if not _check_range(node.get('confidence', 0.5), 0.0, 1.0):
            if node.get('confidence') is not None and 'CONFIDENCE_OUT_OF_RANGE' not in errors:
                errors.append('CONFIDENCE_OUT_OF_RANGE')
        if not _check_range(node.get('static_weight', 0.5), 0.0, 1.0):
            if node.get('static_weight') is not None and 'STATIC_WEIGHT_OUT_OF_RANGE' not in errors:
                errors.append('STATIC_WEIGHT_OUT_OF_RANGE')
        if not _check_range(node.get('health', 0.5), 0.0, 1.0):
            if node.get('health') is not None and 'HEALTH_OUT_OF_RANGE' not in errors:
                errors.append('HEALTH_OUT_OF_RANGE')

    for edge in edges:
        w = edge.get('weight')
        if w is not None and not _check_range(w, 0.0, 1.0):
            if 'EDGE_WEIGHT_OUT_OF_RANGE' not in errors:
                errors.append('EDGE_WEIGHT_OUT_OF_RANGE')

    # ── S1–S4: statistical decision/risk bounds ─────────────────────────────
    for node in nodes:
        prior = node.get('prior_mass')
        if prior is not None and not _check_range(prior, 0.0, 1.0):
            if 'PRIOR_MASS_OUT_OF_RANGE' not in errors:
                errors.append('PRIOR_MASS_OUT_OF_RANGE')
        risk = node.get('risk_bound')
        if risk is not None and not _check_range(risk, 0.0, 1.0):
            if 'RISK_BOUND_OUT_OF_RANGE' not in errors:
                errors.append('RISK_BOUND_OUT_OF_RANGE')

    for edge in edges:
        loss = edge.get('loss_cost')
        if loss is not None and not _check_range(loss, 0.0, 1.0):
            if 'LOSS_COST_OUT_OF_RANGE' not in errors:
                errors.append('LOSS_COST_OUT_OF_RANGE')
            continue
        if loss is not None:
            src_node = node_by_id.get(edge.get('source'))
            tgt_node = node_by_id.get(edge.get('target'))
            src_risk = src_node.get('risk_bound') if src_node else None
            tgt_risk = tgt_node.get('risk_bound') if tgt_node else None
            if src_risk is not None and tgt_risk is not None:
                if float(loss) > float(src_risk) and float(loss) > float(tgt_risk):
                    if 'INADMISSIBLE_VECTOR_RULE' not in errors:
                        errors.append('INADMISSIBLE_VECTOR_RULE')

    # ── D1–D4: differential and dual-comparison graph checks ────────────────
    for row in list(nodes) + list(edges):
        mutation = row.get('mutation')
        if mutation is not None and mutation not in MUTATION_OPS:
            if 'INVALID_MUTATION_OP' not in errors:
                errors.append('INVALID_MUTATION_OP')

    for node in nodes:
        base = node.get('base_version')
        if base is not None and (not isinstance(base, int) or base < 0):
            if 'BASE_VERSION_NEGATIVE' not in errors:
                errors.append('BASE_VERSION_NEGATIVE')

    edge_by_id = {
        edge.get('id'): edge
        for edge in edges
        if isinstance(edge.get('id'), int)
    }
    for edge in edges:
        twin_id = edge.get('dual_twin')
        if twin_id is None:
            continue
        twin = edge_by_id.get(twin_id)
        if twin is None:
            if 'DUAL_TWIN_UNRESOLVED' not in errors:
                errors.append('DUAL_TWIN_UNRESOLVED')
            continue
        edge_id = edge.get('id')
        if edge_id is not None and twin.get('dual_twin') != edge_id:
            if 'DUAL_TWIN_NOT_RECIPROCAL' not in warnings:
                warnings.append('DUAL_TWIN_NOT_RECIPROCAL')

    # ── Q1–Q8: continuous metric / quantum graph / flow field checks ────────
    for node in nodes:
        bc = node.get('boundary_type')
        if bc is not None and bc not in VERTEX_BC:
            if 'INVALID_BOUNDARY_TYPE' not in errors:
                errors.append('INVALID_BOUNDARY_TYPE')

        singularity = node.get('singularity')
        if singularity is not None and singularity not in SINGULARITY_TYPES:
            if 'INVALID_SINGULARITY_TYPE' not in errors:
                errors.append('INVALID_SINGULARITY_TYPE')

        boundary = node.get('text_boundary')
        if boundary is not None and boundary not in TEXT_BOUNDARIES:
            if 'INVALID_TEXT_BOUNDARY' not in errors:
                errors.append('INVALID_TEXT_BOUNDARY')

        for key, err in (
            ('vertex_pot', 'VERTEX_POTENTIAL_NOT_NUMERIC'),
            ('phase_shift', 'PHASE_SHIFT_NOT_NUMERIC'),
            ('divergence', 'DIVERGENCE_NOT_NUMERIC'),
            ('parameter_t', 'CURVE_PARAMETER_NOT_NUMERIC'),
            ('scalar_potential', 'SCALAR_POTENTIAL_NOT_NUMERIC'),
        ):
            value = node.get(key)
            if value is not None and not _is_number(value):
                if err not in errors:
                    errors.append(err)

        for key in ('parameter_generation', 'training_step'):
            value = node.get(key)
            if value is not None and (not isinstance(value, int) or value < 0):
                if key == 'parameter_generation' and 'PARAMETER_GENERATION_NEGATIVE' not in errors:
                    errors.append('PARAMETER_GENERATION_NEGATIVE')
                if key == 'training_step' and 'TRAINING_STEP_NEGATIVE' not in errors:
                    errors.append('TRAINING_STEP_NEGATIVE')

        for key, err in (
            ('loss_value', 'LOSS_VALUE_NEGATIVE'),
            ('grad_norm', 'GRADIENT_NORM_NEGATIVE'),
            ('local_entropy', 'LOCAL_ENTROPY_NEGATIVE'),
            ('entropy_spike', 'ENTROPY_SPIKE_NEGATIVE'),
            ('diffusion_beta_inv', 'DIFFUSION_BETA_INV_NEGATIVE'),
        ):
            value = node.get(key)
            if value is not None and (not _is_number(value) or float(value) < 0.0):
                if err not in errors:
                    errors.append(err)

        for key, err in (
            ('learning_rate', 'LEARNING_RATE_NONPOSITIVE'),
            ('gradient_clip', 'GRADIENT_CLIP_NONPOSITIVE'),
            ('entropy_bound', 'ENTROPY_BOUND_NONPOSITIVE'),
        ):
            value = node.get(key)
            if value is not None and (not _is_number(value) or float(value) <= 0.0):
                if err not in errors:
                    errors.append(err)

        local_entropy = node.get('local_entropy')
        entropy_bound = node.get('entropy_bound')
        if _is_number(local_entropy) and _is_number(entropy_bound):
            if float(local_entropy) > float(entropy_bound):
                if 'ENTROPY_BOUND_EXCEEDED' not in errors:
                    errors.append('ENTROPY_BOUND_EXCEEDED')

        circumradius = node.get('circumradius_R')
        inradius = node.get('inradius_r')
        if circumradius is not None:
            if not _is_number(circumradius) or float(circumradius) < 0.0:
                if 'CIRCUMRADIUS_NEGATIVE' not in errors:
                    errors.append('CIRCUMRADIUS_NEGATIVE')
        if inradius is not None:
            if not _is_number(inradius) or float(inradius) < 0.0:
                if 'INRADIUS_NEGATIVE' not in errors:
                    errors.append('INRADIUS_NEGATIVE')
        if _is_number(circumradius) and _is_number(inradius):
            r_big = float(circumradius)
            r_small = float(inradius)
            if r_big > 0.0 and r_small > 0.0 and r_big < 2.0 * r_small:
                if 'EULER_TRIANGLE_INEQUALITY_BREACH' not in errors:
                    errors.append('EULER_TRIANGLE_INEQUALITY_BREACH')

        arc_length = node.get('arc_length')
        if arc_length is not None and (not _is_number(arc_length) or float(arc_length) < 0.0):
            if 'ARC_LENGTH_NEGATIVE' not in errors:
                errors.append('ARC_LENGTH_NEGATIVE')

        curvature = node.get('curvature')
        if curvature is not None and (not _is_number(curvature) or float(curvature) < 0.0):
            if 'CURVATURE_NEGATIVE' not in errors:
                errors.append('CURVATURE_NEGATIVE')

        rank_p = node.get('tensor_rank_p')
        rank_q = node.get('tensor_rank_q')
        if (rank_p is None) != (rank_q is None):
            if 'TENSOR_RANK_INCOMPLETE' not in errors:
                errors.append('TENSOR_RANK_INCOMPLETE')
        for key in ('tensor_rank_p', 'tensor_rank_q'):
            value = node.get(key)
            if value is not None and (not isinstance(value, int) or value < 0):
                if 'TENSOR_RANK_NEGATIVE' not in errors:
                    errors.append('TENSOR_RANK_NEGATIVE')

        smooth = node.get('is_smooth_section')
        if smooth is not None and smooth not in (0, 1):
            if 'SMOOTH_SECTION_NOT_BOOL' not in errors:
                errors.append('SMOOTH_SECTION_NOT_BOOL')

        curvature_scalar = node.get('riemann_curvature_scalar')
        if curvature_scalar is not None and not _is_number(curvature_scalar):
            if 'CURVATURE_SCALAR_NOT_NUMERIC' not in errors:
                errors.append('CURVATURE_SCALAR_NOT_NUMERIC')

        jacobian_det = node.get('jacobian_determinant')
        if jacobian_det is not None:
            if not _is_number(jacobian_det):
                if 'JACOBIAN_DETERMINANT_NOT_NUMERIC' not in errors:
                    errors.append('JACOBIAN_DETERMINANT_NOT_NUMERIC')
            elif abs(float(jacobian_det)) <= _EPS:
                if 'JACOBIAN_DETERMINANT_ZERO' not in errors:
                    errors.append('JACOBIAN_DETERMINANT_ZERO')

        _validate_tensor_connection(node, errors)
        _validate_lorentz_boost(node, errors)

    for edge in edges:
        field_kind = edge.get('field_kind')
        if field_kind is not None and field_kind not in FIELD_KINDS:
            if 'INVALID_FIELD_KIND' not in errors:
                errors.append('INVALID_FIELD_KIND')

        t0 = edge.get('t0')
        t1 = edge.get('t1')
        if t0 is not None and not _is_number(t0):
            if 'CURVE_PARAMETER_NOT_NUMERIC' not in errors:
                errors.append('CURVE_PARAMETER_NOT_NUMERIC')
        if t1 is not None and not _is_number(t1):
            if 'CURVE_PARAMETER_NOT_NUMERIC' not in errors:
                errors.append('CURVE_PARAMETER_NOT_NUMERIC')
        if _is_number(t0) and _is_number(t1) and float(t1) <= float(t0):
            if 'CURVE_INTERVAL_INVALID' not in errors:
                errors.append('CURVE_INTERVAL_INVALID')

        potential_delta = edge.get('potential_delta')
        if potential_delta is not None and not _is_number(potential_delta):
            if 'POTENTIAL_DELTA_NOT_NUMERIC' not in errors:
                errors.append('POTENTIAL_DELTA_NOT_NUMERIC')

        training_step = edge.get('training_step')
        if training_step is not None and (not isinstance(training_step, int) or training_step < 0):
            if 'TRAINING_STEP_NEGATIVE' not in errors:
                errors.append('TRAINING_STEP_NEGATIVE')

        for key, err in (
            ('loss_value', 'LOSS_VALUE_NEGATIVE'),
            ('grad_norm', 'GRADIENT_NORM_NEGATIVE'),
            ('diffusion_beta_inv', 'DIFFUSION_BETA_INV_NEGATIVE'),
        ):
            value = edge.get(key)
            if value is not None and (not _is_number(value) or float(value) < 0.0):
                if err not in errors:
                    errors.append(err)

        for key, err in (
            ('learning_rate', 'LEARNING_RATE_NONPOSITIVE'),
            ('gradient_clip', 'GRADIENT_CLIP_NONPOSITIVE'),
        ):
            value = edge.get(key)
            if value is not None and (not _is_number(value) or float(value) <= 0.0):
                if err not in errors:
                    errors.append(err)

        for key in ('curl_x', 'curl_y', 'curl_z'):
            value = edge.get(key)
            if value is not None and not _is_number(value):
                if 'CURL_COMPONENT_NOT_NUMERIC' not in errors:
                    errors.append('CURL_COMPONENT_NOT_NUMERIC')
        if field_kind == 'euclidean_gradient':
            curl_values = [edge.get(key) for key in ('curl_x', 'curl_y', 'curl_z')]
            if all(value is not None for value in curl_values) and all(_is_number(value) for value in curl_values):
                if any(abs(float(value)) > _EPS for value in curl_values):
                    if 'EUCLIDEAN_CURL_NONZERO' not in errors:
                        errors.append('EUCLIDEAN_CURL_NONZERO')

        entropy_present = sum(1 for key in ('eh_x', 'eh_y', 'eh_z') if edge.get(key) is not None)
        if 0 < entropy_present < 3:
            if 'PARTIAL_ENTROPY_GRADIENT' not in warnings:
                warnings.append('PARTIAL_ENTROPY_GRADIENT')
        for key in ('eh_x', 'eh_y', 'eh_z', 'entropy_delta'):
            value = edge.get(key)
            if value is not None and not _is_number(value):
                if 'ENTROPY_GRADIENT_NOT_NUMERIC' not in errors:
                    errors.append('ENTROPY_GRADIENT_NOT_NUMERIC')

        length = edge.get('length')
        if length is not None and (not _is_number(length) or float(length) <= 0.0):
            if 'EDGE_LENGTH_NONPOSITIVE' not in errors:
                errors.append('EDGE_LENGTH_NONPOSITIVE')

        refractive = edge.get('refractive')
        if refractive is not None and (not _is_number(refractive) or float(refractive) <= 0.0):
            if 'REFRACTIVE_NONPOSITIVE' not in errors:
                errors.append('REFRACTIVE_NONPOSITIVE')

        potential = edge.get('potential')
        if potential is not None and not _is_number(potential):
            if 'EDGE_POTENTIAL_NOT_NUMERIC' not in errors:
                errors.append('EDGE_POTENTIAL_NOT_NUMERIC')

        frequency = edge.get('frequency')
        if frequency is not None and (not _is_number(frequency) or float(frequency) <= 0.0):
            if 'FREQUENCY_NONPOSITIVE' not in errors:
                errors.append('FREQUENCY_NONPOSITIVE')

        for key, err in (
            ('wave_number', 'WAVE_NUMBER_NOT_NUMERIC'),
            ('amplitude_re', 'AMPLITUDE_NOT_NUMERIC'),
            ('amplitude_im', 'AMPLITUDE_NOT_NUMERIC'),
        ):
            value = edge.get(key)
            if value is not None and not _is_number(value):
                if err not in errors:
                    errors.append(err)

        velocity_present = sum(1 for key in ('vx', 'vy', 'vz') if edge.get(key) is not None)
        if 0 < velocity_present < 3:
            if 'PARTIAL_VELOCITY_VECTOR' not in warnings:
                warnings.append('PARTIAL_VELOCITY_VECTOR')
        for key in ('vx', 'vy', 'vz'):
            value = edge.get(key)
            if value is not None and not _is_number(value):
                if 'VELOCITY_COMPONENT_NOT_NUMERIC' not in errors:
                    errors.append('VELOCITY_COMPONENT_NOT_NUMERIC')

        metric_keys = ('g_00', 'g_01', 'g_10', 'g_11')
        metric_present = sum(1 for key in metric_keys if edge.get(key) is not None)
        if 0 < metric_present < 4:
            if 'PARTIAL_METRIC_TENSOR' not in warnings:
                warnings.append('PARTIAL_METRIC_TENSOR')
        metric_numeric = True
        for key in metric_keys:
            value = edge.get(key)
            if value is not None and not _is_number(value):
                metric_numeric = False
                if 'METRIC_TENSOR_NOT_NUMERIC' not in errors:
                    errors.append('METRIC_TENSOR_NOT_NUMERIC')
        if metric_present == 4 and metric_numeric:
            g00 = float(edge.get('g_00'))
            g01 = float(edge.get('g_01'))
            g10 = float(edge.get('g_10'))
            g11 = float(edge.get('g_11'))
            if abs(g01 - g10) > _EPS:
                if 'METRIC_TENSOR_ASYMMETRIC' not in errors:
                    errors.append('METRIC_TENSOR_ASYMMETRIC')
            if abs((g00 * g11) - (g01 * g10)) <= _EPS:
                if 'METRIC_TENSOR_DEGENERATE' not in errors:
                    errors.append('METRIC_TENSOR_DEGENERATE')

        jacobian_w = edge.get('jacobian_w')
        if jacobian_w is not None:
            if not _is_number(jacobian_w):
                if 'JACOBIAN_WEIGHT_NOT_NUMERIC' not in errors:
                    errors.append('JACOBIAN_WEIGHT_NOT_NUMERIC')
            elif float(jacobian_w) <= 0.0:
                if 'JACOBIAN_WEIGHT_NONPOSITIVE' not in errors:
                    errors.append('JACOBIAN_WEIGHT_NONPOSITIVE')

        _validate_tensor_connection(edge, errors)
        _validate_lorentz_boost(edge, errors)

    # ── G1: finite coordinates ────────────────────────────────────────────────
    for node in nodes:
        for coord in ('x', 'y', 'z'):
            v = node.get(coord)
            if v is not None and isinstance(v, float) and not math.isfinite(v):
                if 'COORDINATE_NOT_FINITE' not in errors:
                    errors.append('COORDINATE_NOT_FINITE')

    # ── G2: partial coordinate vector ─────────────────────────────────────────
    for node in nodes:
        present = sum(1 for c in ('x', 'y', 'z') if node.get(c) is not None)
        if 0 < present < 3:
            if 'PARTIAL_COORDINATE_VECTOR' not in warnings:
                warnings.append('PARTIAL_COORDINATE_VECTOR')

    grade = 'PARSED' if errors else 'VALID'
    return {'grade': grade, 'errors': errors, 'warnings': warnings}


# ── Case checker ─────────────────────────────────────────────────────────────

def _check(result, expected_grade, expected_errors, expected_warnings):
    if result['grade'] != expected_grade:
        return False
    for e in expected_errors:
        if e not in result['errors']:
            return False
    for w in expected_warnings:
        if w not in result['warnings']:
            return False
    # Clean cases: no unexpected issues
    if not expected_errors and not expected_warnings and expected_grade == 'VALID':
        if result['errors'] or result['warnings']:
            return False
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cases = []
    try:
        validator_path = ROOT / 'data/grammar/gml.validator.json'
        validator = json.loads(validator_path.read_text(encoding='utf-8'))
        a1 = validator.get('validation_checks', {}).get('A1', {})
        fatal_errors = set(validator.get('error_taxonomy', {}).get('FATAL', []))
        cases.append((
            'validator_declares_algebraic_interpretation_A1',
            a1.get('layer') == 'algebraic_interpretation_layer' and
            a1.get('name') == 'signature_isomorphism_preserved' and
            a1.get('severity') == 'FATAL' and
            a1.get('error') == 'SIGNATURE_FUNCTOR_FRACTION' and
            'SIGNATURE_FUNCTOR_FRACTION' in fatal_errors
        ))

        fixture_path = ROOT / 'data/conformance/gml_validator.json'
        fixtures = json.loads(fixture_path.read_text(encoding='utf-8'))

        for row in fixtures:
            case_id = row['id']
            stem = row.get('stem', _derive_stem(case_id + '.gml'))
            result = validate_gml(row['gml'], stem)
            passed = _check(result, row['grade'], row['errors'], row['warnings'])
            cases.append((case_id, passed))

        # Shipped GML files — all must be grade VALID with no errors
        for gml_path in sorted((ROOT / 'data').glob('*.gml')):
            stem = _derive_stem(gml_path.name)
            gml_text = gml_path.read_text(encoding='utf-8', errors='replace')
            result = validate_gml(gml_text, stem)
            passed = result['grade'] == 'VALID' and not result['errors']
            cases.append(('shipped/' + gml_path.name, passed))

        return verdict('gml_validator', cases)
    except Exception as exc:
        return verdict('gml_validator', cases, str(exc))


if __name__ == '__main__':
    raise SystemExit(main())
