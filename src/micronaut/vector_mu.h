#pragma once
#include "../world/tensor_substrate.h"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace khanary::micronaut {

struct MicronautContext {
    std::string active_scope;
    float       environmental_temperature = 1.0f;
    uint64_t    trajectory_generation_id  = 0;
};

// Active executable specialist that projects state streamlines across a
// TensorSubstrate field.  Holds epistemic WCR weights (W, C) and a
// numerical projection vector.  Does NOT create or mutate MicronautEntry
// in the registry — VectorMuMicronaut is a projection-layer operator only.
class VectorMuMicronaut {
    uint32_t    micronaut_id_;
    std::string capability_verb_;   // e.g. "@requires", "@related_to"
    float       static_weight_;     // WCR W — structural prior selection weight
    float       confidence_;        // WCR C — epistemic confidence metric
    std::vector<float> numerical_projection_vector_;

public:
    VectorMuMicronaut(uint32_t id, std::string verb, float w, float c)
        : micronaut_id_(id)
        , capability_verb_(std::move(verb))
        , static_weight_(w)
        , confidence_(c) {}

    // Executable operator: μ_v(T) → bounded projection substrate.
    // Returns a new TensorSubstrate initialised from target_substrate metadata;
    // physical_buffer_allocation is null until a concrete GPU backend fills it.
    std::shared_ptr<world::TensorSubstrate> ExecuteProjection(
        const std::shared_ptr<world::TensorSubstrate>& target_substrate,
        const MicronautContext& context)
    {
        auto result = std::make_shared<world::TensorSubstrate>();
        if (target_substrate) {
            result->shape             = target_substrate->shape;
            result->strides           = target_substrate->strides;
            result->dtype             = target_substrate->dtype;
            result->provider_identity = capability_verb_ + ":" +
                                        target_substrate->provider_identity;
        }
        // Concrete GPU dispatch (e.g. geodesic_propagator.hlsl) is wired by
        // a derived class or the FieldDag dispatch method.
        return result;
    }

    void set_projection_vector(std::vector<float> v) {
        numerical_projection_vector_ = std::move(v);
    }

    float       GetStaticWeight() const { return static_weight_; }
    float       GetConfidence()   const { return confidence_; }
    std::string GetVerb()         const { return capability_verb_; }
    uint32_t    GetId()           const { return micronaut_id_; }
    const std::vector<float>& GetProjectionVector() const {
        return numerical_projection_vector_;
    }
};

} // namespace khanary::micronaut
