#include "world/vector_projection.h"
#include "gpu/gram_gpu_scorer.h"

#include <cmath>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

using khanary::world::HyperbolicEventVector;
using khanary::world::HyperbolicAttentionRecord64;
using khanary::world::make_hyperbolic_attention_record;
using khanary::world::pack_hyperbolic_attention_records;

namespace {

bool near_value(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) <= eps;
}

bool check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << "\n";
        return false;
    }
    return true;
}

} // namespace

int main() {
    bool ok = true;

    ok &= check(sizeof(HyperbolicAttentionRecord64) == 64,
                "record_stride_64_bytes");
    ok &= check(alignof(HyperbolicAttentionRecord64) == 16,
                "record_align_16_bytes");

    HyperbolicEventVector timelike;
    timelike.key = "kg:advisor:yax:why_should_i";
    timelike.coords = {1.25f, 0.6f, 0.8f, -0.2f};
    timelike.interval_class = "timelike_candidate";
    timelike.minkowski_interval = -1.042f;

    const auto rec = make_hyperbolic_attention_record(timelike, 0.7f, 0.8f, 0.9f);
    const float space_mag = std::sqrt(0.6f*0.6f + 0.8f*0.8f + 0.2f*0.2f);
    ok &= check(near_value(rec.W, 0.7f) && near_value(rec.C, 0.8f) && near_value(rec.R, 0.9f),
                "wcr_channels_preserved");
    ok &= check(near_value(rec.relation_weight, std::sqrt(0.6f*0.6f + 0.8f*0.8f) / space_mag),
                "relation_weight_from_equatorial_projection");
    ok &= check(near_value(rec.fold_weight, space_mag),
                "fold_weight_from_spatial_magnitude");
    ok &= check(near_value(rec.phase_weight, 0.5f + 0.5f * (-0.2f / space_mag)),
                "phase_weight_from_vertical_projection");
    ok &= check(near_value(rec.tensor_mass, 1.25f) &&
                near_value(rec.tensor_alignment, 1.0f) &&
                near_value(rec.tensor_health, 1.0f) &&
                near_value(rec.tensor_enabled, 1.0f),
                "tensor_slot_projection_channels");
    ok &= check(near_value(rec.reserved0, 0.6f) &&
                near_value(rec.reserved1, 0.8f) &&
                near_value(rec.reserved2, -0.2f),
                "raw_hyperbolic_spatial_coords_carried_in_reserved_fields");

    HyperbolicEventVector spacelike = timelike;
    spacelike.key = "kg:advisor:yax:spacelike";
    spacelike.interval_class = "spacelike_candidate";
    const auto blocked = make_hyperbolic_attention_record(spacelike, 2.0f, -1.0f, 0.25f);
    ok &= check(near_value(blocked.W, 1.0f) &&
                near_value(blocked.C, 0.0f) &&
                near_value(blocked.R, 0.25f),
                "wcr_channels_clamped");
    ok &= check(near_value(blocked.tensor_alignment, 0.0f),
                "spacelike_event_has_zero_causal_alignment");

    std::unordered_map<std::string, float> w{{timelike.key, 0.33f}};
    std::unordered_map<std::string, float> c{{timelike.key, 0.44f}};
    std::unordered_map<std::string, float> r{{timelike.key, 0.55f}};
    std::vector<HyperbolicEventVector> events{timelike, spacelike};
    const auto packed = pack_hyperbolic_attention_records(events, w, c, r);
    ok &= check(packed.size() == 2, "packed_record_count");
    ok &= check(near_value(packed[0].W, 0.33f) &&
                near_value(packed[0].C, 0.44f) &&
                near_value(packed[0].R, 0.55f),
                "map_overrides_applied");
    ok &= check(near_value(packed[1].W, 1.0f) &&
                near_value(packed[1].C, 1.0f) &&
                near_value(packed[1].R, 0.5f),
                "map_defaults_applied");

    const auto boost = khanary::gpu::LorentzBoostRows::make_pure(
        0.5f, {1.0f, 0.0f, 0.0f});
    const float ch = std::cosh(0.5f);
    const float sh = std::sinh(0.5f);
    ok &= check(boost.enabled, "lorentz_make_pure_enabled");
    ok &= check(near_value(boost.row0[0], ch) && near_value(boost.row0[1], sh) &&
                near_value(boost.row1[0], sh) && near_value(boost.row1[1], ch),
                "lorentz_make_pure_x_boost_rows");
    const float u0 = boost.row0[0] * 1.0f;
    const float u1 = boost.row1[0] * 1.0f;
    ok &= check(near_value(-u0 * u0 + u1 * u1, -1.0f, 1e-4f),
                "lorentz_make_pure_preserves_minkowski_interval");
    const auto invalid = khanary::gpu::LorentzBoostRows::make_pure(
        0.5f, {0.0f, 0.0f, 0.0f});
    ok &= check(!invalid.enabled, "lorentz_make_pure_rejects_zero_direction");

    if (!ok) return 1;
    std::cout << "hyperbolic attention packing PASS\n";
    return 0;
}

