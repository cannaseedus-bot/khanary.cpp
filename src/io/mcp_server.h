#pragma once
#include "json_rpc_server.h"
#include "../manifest/manifest_registry.h"
#include "../kuhul/kuhul_vm.h"
#include "../micronaut/micronaut_registry.h"
#include "../micronaut/factory.h"
#include "../micronaut/sandbox_inspector.h"
#include "../memory/confidence.h"
#include "../world/world_state.h"
#include "../world/field_dag.h"
#include "jrom.h"
#include <string>
#include <memory>

namespace khanary::io {

// MCP gateway on port 8764
// Exposes 16 tools: kuhul_forge, manifest_resolve, fold_execute, micronaut_run,
//   grammar_check, phase_query, gram_read, gram_write, confidence_get, jrom_replay,
//   sandbox_query, sandbox_list, sandbox_violations, world_state, field_dag, field_dag_svg
class MCPServer {
public:
    MCPServer(
        uint16_t port,
        manifest::ManifestRegistry& reg,
        kuhul::KuhulVM& vm,
        micronaut::MicronautRegistry& mu_reg,
        micronaut::Factory& factory,
        micronaut::SandboxInspector& sandbox,
        memory::ConfidenceStore& confidence,
        world::WorldState& world,
        JROM& jrom);

    bool start();
    void stop();
    bool is_running() const;

private:
    JsonRpcServer rpc_;

    manifest::ManifestRegistry&     manifest_;
    kuhul::KuhulVM&                  vm_;
    micronaut::MicronautRegistry&    mu_reg_;
    micronaut::Factory&              factory_;
    micronaut::SandboxInspector&     sandbox_;
    memory::ConfidenceStore&         confidence_;
    world::WorldState&               world_;
    JROM&                            jrom_;

    void register_tools();

    // Tool implementations
    util::JsonPtr tool_kuhul_forge(const util::JsonPtr& params);
    util::JsonPtr tool_manifest_resolve(const util::JsonPtr& params);
    util::JsonPtr tool_fold_execute(const util::JsonPtr& params);
    util::JsonPtr tool_micronaut_run(const util::JsonPtr& params);
    util::JsonPtr tool_grammar_check(const util::JsonPtr& params);
    util::JsonPtr tool_phase_query(const util::JsonPtr& params);
    util::JsonPtr tool_gram_read(const util::JsonPtr& params);
    util::JsonPtr tool_gram_write(const util::JsonPtr& params);
    util::JsonPtr tool_confidence_get(const util::JsonPtr& params);
    util::JsonPtr tool_jrom_replay(const util::JsonPtr& params);
    // Sandbox inspection
    util::JsonPtr tool_sandbox_query(const util::JsonPtr& params);
    util::JsonPtr tool_sandbox_list(const util::JsonPtr& params);
    util::JsonPtr tool_sandbox_violations(const util::JsonPtr& params);
    // K-CUBE 3D world state
    util::JsonPtr tool_world_state(const util::JsonPtr& params);
    // 2D semantic field DAG → XML (topology + tensor weights)
    util::JsonPtr tool_field_dag(const util::JsonPtr& params);
    // 2D field graph + SVG-3D isometric tensor blocks → SVG string
    util::JsonPtr tool_field_dag_svg(const util::JsonPtr& params);
};

} // namespace khanary::io
