#pragma once

#include "d3d11_device.h"
#include "compute_dispatch.h"
#include "../world/field_dag.h"
#include <string>

namespace khanary::gpu {

class SvgComputeRuntime {
public:
    explicit SvgComputeRuntime(D3D11Device& dev);

    // Compile and bind node/edge projection entry points from data/shaders/svg_compute.hlsl.
    bool init(const std::string& shader_path);

    bool is_ready() const { return ready_; }
    const std::string& last_error() const { return last_error_; }

    // Project FieldDag tensor state into screen-space geometry/state channels.
    // Score channels follow KHANARY W·C·R semantics where available.
    bool project(
        const std::vector<world::FieldNode>& nodes,
        const std::vector<world::FieldEdge>& edges,
        world::FieldSvgProjection& out_projection);

private:
    D3D11Device&   dev_;
    ComputeDispatch node_dispatch_;
    ComputeDispatch edge_dispatch_;
    bool            ready_ = false;
    std::string     shader_path_;
    std::string     last_error_;

    bool compile_shader_entry(
        const std::string& source,
        const char* entry,
        ComputeDispatch& dispatch,
        std::string* error_out);
};

} // namespace khanary::gpu
