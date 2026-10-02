"""Validate the dataset ingestion bootstrap artifacts.

Suite name: dataset_ingestion  (must match run_conformance.py required set)
"""

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))
from conformance_common import verdict  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


REQUIRED_CHANNELS = {
    "unigram",
    "bigram",
    "trigram",
    "quadgram",
    "coarse_gram",
    "function_gram",
    "tool_gram",
}

EXPECTED_TRACK_SLUGS = {"joke_mu", "mx2lm", "advisor_mu", "plan_mu"}


def load_json(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def load_jsonl(path):
    rows = []
    with (ROOT / path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    cases = []
    try:
        source_manifest = load_json("data/ingest/khanary_dataset_sources.json")
        ingest_manifest = load_json("data/ingest/dataset_ingestion_manifest.json")

        cases.append(("source_manifest_declares_boundary",
                      source_manifest.get("@kind") == "khanary.dataset_sources.v1" and
                      "authority_boundary" in source_manifest and
                      len(source_manifest.get("sources", [])) >= 49))
        sources = source_manifest.get("sources", [])
        cases.append(("source_manifest_has_authority_and_candidate_sources",
                      any(row.get("semantic_authority") is True for row in sources if isinstance(row, dict)) and
                      any(row.get("semantic_authority") is not True for row in sources if isinstance(row, dict))))

        policy = ingest_manifest.get("policy", {})
        cases.append(("ingest_manifest_policy_is_candidate_only",
                      ingest_manifest.get("@type") == "khanary.dataset_ingestion_manifest.v1" and
                      policy.get("raw_dataset_copy") is False and
                      policy.get("semantic_authority") is False and
                      policy.get("vector_projection") == "derived_rebuildable" and
                      policy.get("gram_pools") == "per_track"))

        totals = ingest_manifest.get("totals", {})
        cases.append(("ingest_manifest_has_streamed_rows",
                      totals.get("sources_read", 0) >= 1 and
                      totals.get("rows_used", 0) >= 1 and
                      totals.get("lexical_units", 0) >= totals.get("rows_used", 0)))
        cases.append(("ingest_manifest_uses_lexical_units_not_model_tokens",
                      "lexical_units" in totals and "tokens" not in totals and
                      all("lexical_units" in row and "tokens" not in row
                          for row in ingest_manifest.get("source_summaries", [])
                          if row.get("read_mode") == "streamed")))
        cases.append(("authority_sources_reference_only_by_default",
                      totals.get("authority_sources_referenced", 0) >= 1 and
                      all(row.get("read_mode") == "authority_reference_only"
                          for row in ingest_manifest.get("source_summaries", [])
                          if row.get("semantic_authority") is True)))

        tracks = ingest_manifest.get("tracks", [])
        slugs = {row.get("slug") for row in tracks if isinstance(row, dict)}
        cases.append(("per_track_pools_declared",
                      isinstance(tracks, list) and len(tracks) >= 4 and
                      EXPECTED_TRACK_SLUGS.issubset(slugs)))

        total_candidate_rows = 0
        total_projection_rows = 0
        total_track_rows = 0
        total_track_lexical_units = 0
        all_tracks_valid = True
        all_track_artifacts_exist = True
        all_track_binary_valid = True
        all_track_counts_match = True
        all_track_candidates_valid = True
        all_track_vectors_valid = True
        all_track_pattern_links_valid = True
        all_track_summaries_distinguish_counts = True

        for track in tracks:
            if not isinstance(track, dict):
                all_tracks_valid = False
                continue
            track_name = str(track.get("track", ""))
            slug = str(track.get("slug", ""))
            artifacts = track.get("artifacts", {})
            if not isinstance(artifacts, dict):
                all_tracks_valid = False
                continue

            required_artifact_keys = {
                "gram_container",
                "pattern_container",
                "vector_projection_container",
                "ngram_summary",
                "candidate_key_grammar",
                "binary_snapshot",
            }
            if not required_artifact_keys.issubset(artifacts):
                all_track_artifacts_exist = False
                continue

            paths = {key: ROOT / value for key, value in artifacts.items()}
            if not all(paths[key].exists() for key in required_artifact_keys):
                all_track_artifacts_exist = False
                continue

            gram = json.loads(paths["gram_container"].read_text(encoding="utf-8"))
            pattern = json.loads(paths["pattern_container"].read_text(encoding="utf-8"))
            vector = json.loads(paths["vector_projection_container"].read_text(encoding="utf-8"))
            summary = json.loads(paths["ngram_summary"].read_text(encoding="utf-8"))
            candidates = load_jsonl(artifacts["candidate_key_grammar"])

            channels = gram.get("channels", {})
            channel_counts = gram.get("channel_counts", {})
            if not (
                gram.get("@type") == "mx2lm.gram_container.v1" and
                gram.get("key") == f"mx2lm:dataset_gram_container:{slug}:v1" and
                gram.get("track") == track_name and
                gram.get("semantic_authority") is False and
                gram.get("admission") == "candidate" and
                REQUIRED_CHANNELS.issubset(channels)
            ):
                all_tracks_valid = False

            if not (
                all(channel_counts.get(name) == len(channels.get(name, []))
                    for name in REQUIRED_CHANNELS) and
                gram.get("total_grams") == sum(channel_counts.get(name, 0)
                                               for name in REQUIRED_CHANNELS) and
                track.get("candidate_rows") == gram.get("total_grams")
            ):
                all_track_counts_match = False

            observed_counts = summary.get("observed_unique_channel_counts", {})
            emitted_counts = summary.get("emitted_channel_counts", {})
            if not (
                summary.get("@type") == "mx2lm.dataset_ngram_summary.v1" and
                summary.get("track") == track_name and
                summary.get("semantic_authority") is False and
                summary.get("admission") == "candidate" and
                emitted_counts == channel_counts and
                summary.get("total_grams") == gram.get("total_grams") and
                summary.get("observed_total_unique_grams", 0) >= summary.get("total_grams", 0) and
                all(observed_counts.get(name, 0) >= channel_counts.get(name, 0)
                    for name in REQUIRED_CHANNELS)
            ):
                all_track_summaries_distinguish_counts = False

            if not (
                pattern.get("@type") == "mx2lm.pattern_container.v1" and
                pattern.get("key") == f"mx2lm:dataset_pattern_container:{slug}:v1" and
                pattern.get("track") == track_name and
                pattern.get("semantic_authority") is False and
                pattern.get("events", [{}])[0].get("gram_container") == gram.get("key")
            ):
                all_track_pattern_links_valid = False

            if not (
                len(candidates) == gram.get("total_grams") and
                all(row.get("@type") == "khanary.key_grammar.candidate.v1" and
                    row.get("semantic_authority") is False and
                    row.get("admission") == "candidate" and
                    row.get("target_container") == "mx2lm:gram_container:v1" and
                    isinstance(row.get("key_book"), list) and
                    row.get("key", "").startswith("dataset:")
                    for row in candidates)
            ):
                all_track_candidates_valid = False

            projections = vector.get("projections", [])
            candidate_keys = {row.get("key") for row in candidates}
            if not (
                vector.get("@type") == "vector.projection_container.v1" and
                vector.get("namespace") == "VECTOR-\u00b5" and
                vector.get("semantic_authority") is False and
                vector.get("responsibility") == "derived_geometry_projection_only" and
                vector.get("projection_count") == len(projections) == len(candidates) and
                all(row.get("semantic_authority") is False and
                    row.get("source_key") in candidate_keys and
                    row.get("projection_track") == "VECTOR-\u00b5"
                    for row in projections)
            ):
                all_track_vectors_valid = False

            binary_info = track.get("binary_snapshot", {})
            if not (
                paths["binary_snapshot"].exists() and
                paths["binary_snapshot"].stat().st_size > 24 and
                binary_info.get("magic") == "MX2NGRAM" and
                binary_info.get("sections") == binary_info.get("emitted_section_counts")
            ):
                all_track_binary_valid = False

            total_candidate_rows += int(track.get("candidate_rows", 0) or 0)
            total_projection_rows += int(track.get("projection_count", 0) or 0)
            total_track_rows += int(track.get("totals", {}).get("rows_used", 0) or 0)
            total_track_lexical_units += int(track.get("totals", {}).get("lexical_units", 0) or 0)

        cases.append(("per_track_artifacts_exist", all_track_artifacts_exist))
        cases.append(("per_track_gram_containers_valid", all_tracks_valid))
        cases.append(("per_track_gram_counts_match", all_track_counts_match))
        cases.append(("per_track_summaries_distinguish_observed_from_emitted",
                      all_track_summaries_distinguish_counts))
        cases.append(("per_track_pattern_links_valid", all_track_pattern_links_valid))
        cases.append(("per_track_candidate_key_grammar_valid", all_track_candidates_valid))
        cases.append(("per_track_vector_projection_valid", all_track_vectors_valid))
        cases.append(("per_track_binary_snapshots_valid", all_track_binary_valid))
        cases.append(("per_track_totals_match_manifest_totals",
                      total_track_rows == totals.get("rows_used", 0) and
                      total_track_lexical_units == totals.get("lexical_units", 0) and
                      total_candidate_rows >= len(tracks) and
                      total_projection_rows == total_candidate_rows))

        return verdict("dataset_ingestion", cases)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        cases.append(("dataset_ingestion_exception", False))
        return verdict("dataset_ingestion", cases, error=str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
