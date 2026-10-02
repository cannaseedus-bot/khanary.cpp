#pragma once
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace khanary::gpu {

// STBMap — Semantic Tensor Buffer Map.
//
// Binary storage format for Q/K/V semantic tensor state, parallel to SDDS.
// Where SDDS stores raw weight shards, STBMap stores the derived semantic
// state: node attention vectors, fold weights, gram weights, geodesic geometry.
//
// Wire format (all values little-endian):
//   [0..3]   magic "STBM"
//   [4..7]   version = 1
//   [8..11]  section_count
//   Per section:
//     [0..3]   type (STBSectionType as uint32)
//     [4..7]   record_count
//     [8..11]  data_bytes
//     [12..15] crc32
//     data (record_count × sizeof_per_type)

enum class STBSectionType : uint32_t {
    Field    = 0,  // per-node Q/K/V attention vectors
    Fold     = 1,  // per-edge fold weights
    Gram     = 2,  // per-gram entry weights
    Geodesic = 3,  // per-node entropic geometry state
};

// Field record: Q(3) + K(3) + V(3) = 9 × float32 = 36 bytes
struct STBFieldRecord {
    float q[3];  // query direction (phase unit vector)
    float k[3];  // key direction
    float v[3];  // value triple: entropy, weight, confidence
};

// Fold record: phase indices + geometry = 3 × uint32 + 2 × float = 20 bytes
struct STBFoldRecord {
    uint32_t from_phase;   // 0..5 (C6 phase index)
    uint32_t to_phase;     // 0..5
    uint32_t fold_type;    // 0 = horizontal, 1 = vertical
    float    arc_weight;   // cos(Δθ)
    float    delta_theta;  // degrees
};

// Gram record: null-padded key + 3 weights = 48 + 3 × float = 60 bytes
struct STBGramRecord {
    char  key[48];    // null-padded gram key string (max 47 usable chars)
    float q_weight;
    float k_weight;
    float v_weight;
};

// Geodesic record: position + entropy/mesh field = 9 × float32 = 36 bytes.
//
// This is the optional sidecar geometry consumed by attention.hlsl at t1.
// The main SemanticRecord stays 64 bytes; these richer fields are opt-in
// projection modifiers and do not carry semantic authority.
struct STBGeodesicRecord {
    float x, y, z;
    float local_entropy;
    float geodesic_entropy;
    float unruh_temp;
    float divergence;
    float circumradius_R;
    float inradius_r;
};

struct STBSection {
    STBSectionType       type;
    uint32_t             record_count = 0;
    std::vector<uint8_t> data;
};

struct STBMap {
    std::vector<STBSection> sections;
};

// ── Builders: pack typed records into an STBSection ───────────────────────

STBSection make_field_section(const std::vector<STBFieldRecord>& records);
STBSection make_fold_section(const std::vector<STBFoldRecord>& records);
STBSection make_gram_section(const std::vector<STBGramRecord>& records);
STBSection make_geodesic_section(const std::vector<STBGeodesicRecord>& records);

// ── Extractors: unpack typed records from an STBSection ──────────────────

std::vector<STBFieldRecord>    get_field_records(const STBSection& sec);
std::vector<STBFoldRecord>     get_fold_records(const STBSection& sec);
std::vector<STBGramRecord>     get_gram_records(const STBSection& sec);
std::vector<STBGeodesicRecord> get_geodesic_records(const STBSection& sec);

// ── I/O ──────────────────────────────────────────────────────────────────

// Write map to file.  Throws std::runtime_error on I/O failure.
void write_stb_map(const STBMap& map, const std::string& path);

// Read and CRC-validate map from file.  Throws on I/O error or corruption.
STBMap read_stb_map(const std::string& path);

} // namespace khanary::gpu
