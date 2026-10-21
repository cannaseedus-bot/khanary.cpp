# MX2LM Vector Field

The MX2LM vector field is a corpus-probability geometric graph stored in SafeTensors format. It converts `tools/mx2lm.arpa` (output of `train_srilm.bat`) into a queryable three-layer field where nodes are n-grams positioned in unit space and edge weights are ARPA conditional probabilities.

```text
mx2lm.arpa  (SRILM output)
  -> arpa_to_safetensors.py
  -> data/mx2lm.vectors.safetensors   153 MB   4.3M nodes  3.5M edges
     data/mx2lm.vectors.labels.json    82 MB   index -> surface string
     data/mx2lm.vectors.meta.json              counts + phase map
```

This is the **W_corpus** component of WCR scoring — frozen corpus prior baked at generation time. W_learned (gram_store evidence) overlays it dynamically; W×C×R improves without regenerating the files.

---

## Tensor schema

| Tensor | Shape | dtype | Meaning |
|---|---|---|---|
| `node_positions` | `[N, 3]` | float32 | `(x, y, z)` — FNV-1a hash → (x,y) ∈ [0,1]², phase → z |
| `node_weights` | `[N]` | float32 | Static corpus probability P(ngram) |
| `node_phase` | `[N]` | int32 | Phase index: 0=Pop 1=Wo 2=Yax |
| `edge_index` | `[2, E]` | int64 | COO directed edges: `[0,:]`=src `[1,:]`=tgt |
| `edge_weights` | `[E]` | float32 | Conditional P(wn \| context) — ARPA log_prob directly |

Coordinates: `x = (fnv1a(label) & 0xFFFF) / 65535.0`, `y = ((fnv1a(label) >> 16) & 0xFFFF) / 65535.0`, `z` = phase constant. Positions are geometric addresses, not semantic embeddings — proximity is hash-coincidence, not meaning.

---

## Three-layer structure

Each ngram width maps to a K'UHUL phase:

| Phase | z | Layer | `node_weight` | `edge_weight` |
|---|---|---|---|---|
| **Pop** | 0.000 | unigrams | P(w) | — |
| **Wo** | 0.167 | bigrams | P(w₁,w₂) = P(w₁)×P(w₂\|w₁) | P(w₂\|w₁) |
| **Yax** | 0.333 | trigrams | P(w₁,w₂,w₃) via chain rule | P(w₃\|w₁,w₂) |

**Key invariant**: ARPA log_prob for bigrams/trigrams is already the conditional P(wn|context). Use it directly as the edge weight. Node joint probabilities are computed via chain rule — do not divide the ARPA value by the marginal (that inflates weights past 1.0).

Edges connect same-phase nodes: Pop→Wo (unigram to bigram) and Wo→Yax (bigram to trigram). The edge weight is the conditional that extends the context.

---

## Regenerating

```powershell
# SafeTensors (Python/ML path — recommended)
py -3 tools\arpa_to_safetensors.py tools\mx2lm.arpa --min-prob 1e-6

# GML overlay (KHANARY C++ native path — FieldDag::load_gml)
py -3 tools\arpa_to_gml_overlay.py tools\mx2lm.arpa --min-prob 1e-6
```

`--min-prob 1e-6` drops ~567k hapax unigrams (Zipf tail, P < 1 in a million tokens) while keeping all bigrams/trigrams (already pruned by SRILM `-gtmin 2`).

Stats at `--min-prob 1e-6`:

| | |
|---|---|
| Unigrams | 31,760 |
| Bigrams | 2,665,260 |
| Trigrams | 1,603,960 |
| Total nodes | 4,300,980 |
| Total edges | 3,542,274 |
| SafeTensors size | 153 MB |
| GML size | 1.66 GB |

---

## Querying with field_query.py

XQuery-style declarative path navigation over the loaded field:

```
Phase::selector[@predicate] / Phase::selector[@predicate] ...
```

Phase axes: `Pop` (unigrams), `Wo` (bigrams), `Yax` (trigrams). `*` is wildcard selector. `/` steps along out-edges.

Predicates:
- `@weight` — node joint probability P(ngram)
- `@cond` — incoming edge conditional weight P(wn|context)
- `@label` — surface string, ops: `=`, `!=`, `contains`, `starts-with`
- `@x`, `@y`, `@z` — geometric position

Score of a result path = product of traversed edge conditionals = P(chain | seed).

```powershell
# REPL
py -3 tools\field_query.py

# One-shot
py -3 tools\field_query.py "Pop::kuhul / Wo::*[@cond > 0.3] / Yax::*"
```

Example queries:

```
Pop::kuhul / Wo::*                       all bigrams starting with "kuhul"
Pop::the / Wo::*[@cond > 0.4] / Yax::*  high-prob trigrams starting with "the"
Pop::*[@weight > 0.005]                  common unigrams
Wo::*[@label starts-with "of "]         bigrams prefixed by "of "
```

---

## Question-to-Fold compiler

`tools/question_to_fold.py` translates a natural language question into a compiled question record (`kh.question-fold/1`) that specifies the XCFE intent class, field path, and WCR ranking policy.

Five operators, each mapping to an XCFE class and K'UHUL phase:

| Operator | XCFE class | Phase | Trigger patterns |
|---|---|---|---|
| DEFINE | CONDITION + VERIFICATION | Wo / Ch'en | "What is X?" "Define X" |
| EXPLAIN | CAUSAL | Wo | "How does X work?" "Explain X" |
| COMPARE | SELECTION | Yax | "Compare X and Y" "X vs Y" |
| WHY | CAUSAL | Wo | "Why is X?" "Why are X?" |
| DISCOVER | EXECUTION | Sek | "What connects X and Y?" "Find X" |

```powershell
# Compile only — print JSON record
py -3 tools\question_to_fold.py "What is a fold?"

# Compile + execute field path
py -3 tools\question_to_fold.py --run "What is kuhul"

# REPL
py -3 tools\question_to_fold.py
```

Compiled record schema (`kh.question-fold/1`):

```json
{
  "schema": "kh.question-fold/1",
  "question": "What is a fold?",
  "intent": "DEFINE",
  "target": "fold",
  "xcfe": {
    "class": "CONDITION",
    "phase": "Wo",
    "verification_class": "VERIFICATION",
    "verification_phase": "Ch'en"
  },
  "field": {
    "path": "Pop::fold / Wo::*[@cond > 0.3] / Yax::*",
    "threshold": 0.3,
    "limit": 100
  },
  "ranking": {
    "prior": "safetensors",
    "learned": "gram_store",
    "relevance": "definition",
    "combine": "multiply"
  },
  "learning": {
    "read_phase": "Pop",
    "write_phase": "Xul",
    "write_requires": "evaluated_outcome"
  }
}
```

The compiler is a grammar frontend for XCFE. It does not execute folds, grant authority, or write to gram_store — the `write_requires: evaluated_outcome` gate means gramWrite only fires after Ch'en verification of the retrieved answer.

---

## WCR integration

The field provides the **W_corpus** (static prior) component of WCR scoring:

```
S(k | q, c) = W_corpus(k) × W_learned(k) × C(k, c) × R(k, q)
```

- `W_corpus(k)` = `node_weights[k]` from safetensors — frozen corpus probability
- `W_learned(k)` = gram_store evidence weight — mutable, defaults to 1.0 (neutral) for unseen keys
- `C(k, c)` = contextual confidence — runtime confidence given the current context
- `R(k, q)` = query relevance — varies by intent (definition depth, causal chain, etc.)

The W_effective split (`W_corpus × W_learned`) keeps the immutable ARPA prior separate from the mutable learned overlay. New evidence from gram_store raises C dynamically; the safetensors file does not need to be regenerated.

---

## Loading in Python

```python
from safetensors.numpy import load_file
import json

d      = load_file("data/mx2lm.vectors.safetensors")
labels = json.loads(open("data/mx2lm.vectors.labels.json").read())

d["node_positions"]   # float32 [N, 3]  — (x, y, z) coordinates
d["node_weights"]     # float32 [N]     — P(ngram)
d["node_phase"]       # int32   [N]     — 0=Pop 1=Wo 2=Yax
d["edge_index"]       # int64   [2, E]  — COO edges
d["edge_weights"]     # float32 [E]     — P(wn | context)
```
