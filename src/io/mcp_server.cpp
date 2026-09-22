#include "mcp_server.h"
#include "../util/json.h"

namespace khanary::io {

using namespace khanary::util;

MCPServer::MCPServer(
    uint16_t port,
    manifest::ManifestRegistry& reg,
    kuhul::KuhulVM& vm,
    micronaut::MicronautRegistry& mu_reg,
    micronaut::Factory& factory,
    micronaut::SandboxInspector& sandbox,
    memory::ConfidenceStore& confidence,
    world::WorldState& world,
    JROM& jrom)
    : rpc_(port), manifest_(reg), vm_(vm), mu_reg_(mu_reg),
      factory_(factory), sandbox_(sandbox), confidence_(confidence),
      world_(world), jrom_(jrom)
{
    register_tools();
}

bool MCPServer::start()    { return rpc_.start(); }
void MCPServer::stop()     { rpc_.stop(); }
bool MCPServer::is_running() const { return rpc_.is_running(); }

void MCPServer::register_tools() {
    rpc_.register_method("kuhul_forge",         [this](const JsonPtr& p){ return tool_kuhul_forge(p); });
    rpc_.register_method("manifest_resolve",    [this](const JsonPtr& p){ return tool_manifest_resolve(p); });
    rpc_.register_method("fold_execute",        [this](const JsonPtr& p){ return tool_fold_execute(p); });
    rpc_.register_method("micronaut_run",       [this](const JsonPtr& p){ return tool_micronaut_run(p); });
    rpc_.register_method("grammar_check",       [this](const JsonPtr& p){ return tool_grammar_check(p); });
    rpc_.register_method("phase_query",         [this](const JsonPtr& p){ return tool_phase_query(p); });
    rpc_.register_method("gram_read",           [this](const JsonPtr& p){ return tool_gram_read(p); });
    rpc_.register_method("gram_write",          [this](const JsonPtr& p){ return tool_gram_write(p); });
    rpc_.register_method("confidence_get",      [this](const JsonPtr& p){ return tool_confidence_get(p); });
    rpc_.register_method("jrom_replay",         [this](const JsonPtr& p){ return tool_jrom_replay(p); });
    rpc_.register_method("sandbox_query",       [this](const JsonPtr& p){ return tool_sandbox_query(p); });
    rpc_.register_method("sandbox_list",        [this](const JsonPtr& p){ return tool_sandbox_list(p); });
    rpc_.register_method("sandbox_violations",  [this](const JsonPtr& p){ return tool_sandbox_violations(p); });
    rpc_.register_method("world_state",         [this](const JsonPtr& p){ return tool_world_state(p); });
    rpc_.register_method("field_dag",            [this](const JsonPtr& p){ return tool_field_dag(p); });
    rpc_.register_method("field_dag_svg",        [this](const JsonPtr& p){ return tool_field_dag_svg(p); });
}

JsonPtr MCPServer::tool_kuhul_forge(const JsonPtr& params) {
    if (!params || !params->is_object()) return json_string("error: params required");
    auto prog = params->get_string("program");
    if (!prog) return json_string("error: program required");
    kuhul::XCFEContext ctx;
    auto result = vm_.run(*prog, ctx);
    JsonObject obj;
    obj["ok"]    = json_bool(result.ok);
    obj["error"] = json_string(result.error);
    return json_object(std::move(obj));
}

JsonPtr MCPServer::tool_manifest_resolve(const JsonPtr& params) {
    if (!params || !params->is_object()) return json_null();
    auto key = params->get_string("key");
    if (!key) return json_null();
    auto expanded = manifest_.resolve(*key);
    JsonArray arr;
    for (const auto& s : expanded) arr.push_back(json_string(s));
    return json_array(std::move(arr));
}

JsonPtr MCPServer::tool_fold_execute(const JsonPtr& params) {
    if (!params || !params->is_object()) return json_null();
    auto prog = params->get_string("program");
    if (!prog) return json_null();
    auto result = vm_.run(*prog);
    JsonObject obj;
    obj["ok"] = json_bool(result.ok);
    obj["error"] = json_string(result.error);
    return json_object(std::move(obj));
}

JsonPtr MCPServer::tool_micronaut_run(const JsonPtr& params) {
    if (!params || !params->is_object()) return json_null();
    auto id = params->get_string("id");
    if (!id) return json_null();
    const auto* e = mu_reg_.get(*id);
    if (!e) return json_string("not found");
    JsonObject obj;
    obj["id"]    = json_string(e->id);
    obj["phase"] = json_string(kuhul::phase_name(e->primary_phase));
    obj["active"]= json_bool(e->active);
    JsonArray caps;
    for (const auto& c : e->capabilities) caps.push_back(json_string(c));
    obj["capabilities"] = json_array(std::move(caps));
    return json_object(std::move(obj));
}

JsonPtr MCPServer::tool_grammar_check(const JsonPtr& params) {
    if (!params || !params->is_object()) return json_null();
    auto prog = params->get_string("program");
    if (!prog) return json_null();
    auto instructions = kuhul::KuhulVM::parse_program(*prog);
    JsonObject obj;
    obj["valid"]     = json_bool(!instructions.empty());
    obj["instr_count"] = json_number((double)instructions.size());
    return json_object(std::move(obj));
}

JsonPtr MCPServer::tool_phase_query(const JsonPtr& params) {
    if (!params || !params->is_object()) return json_null();
    auto phase_s = params->get_string("phase");
    if (!phase_s) return json_null();
    kuhul::Phase p;
    if (!kuhul::phase_from_name(*phase_s, p)) return json_string("unknown phase");
    JsonObject obj;
    obj["name"]   = json_string(kuhul::phase_name(p));
    obj["angle"]  = json_number(kuhul::phase_angle(p));
    obj["next"]   = json_string(kuhul::phase_name(kuhul::next_phase(p)));
    return json_object(std::move(obj));
}

JsonPtr MCPServer::tool_gram_read(const JsonPtr& params) {
    if (!params || !params->is_object()) return json_null();
    auto key = params->get_string("key");
    if (!key) return json_null();
    auto g = vm_.gram_store().gramRead(*key);
    if (!g) return json_null();
    JsonObject obj;
    obj["key"]   = json_string(g->key);
    obj["value"] = json_string(g->value);
    obj["phase"] = json_string(g->phase);
    return json_object(std::move(obj));
}

JsonPtr MCPServer::tool_gram_write(const JsonPtr& params) {
    if (!params || !params->is_object()) return json_null();
    auto key = params->get_string("key");
    auto val = params->get_string("value");
    if (!key || !val) return json_bool(false);
    vm_.gram_store().gramWrite(*key, *val, "Xul");
    jrom_.log("gram_write", "{\"key\":\"" + *key + "\"}");
    return json_bool(true);
}

JsonPtr MCPServer::tool_confidence_get(const JsonPtr& params) {
    if (!params || !params->is_object()) return json_null();
    auto id = params->get_string("id");
    if (!id) return json_null();
    return json_number(confidence_.get(*id));
}

JsonPtr MCPServer::tool_jrom_replay(const JsonPtr& params) {
    uint64_t from_seq = 0;
    if (params && params->is_object()) {
        if (auto n = params->get_number("from_seq")) from_seq = (uint64_t)*n;
    }
    JsonArray events;
    jrom_.replay(from_seq, [&](const JROMEvent& e) {
        JsonObject obj;
        obj["seq"]  = json_number((double)e.seq);
        obj["type"] = json_string(e.type);
        events.push_back(json_object(std::move(obj)));
    });
    return json_array(std::move(events));
}

// ── Sandbox inspection tools ──────────────────────────────────────────────────

JsonPtr MCPServer::tool_sandbox_query(const JsonPtr& params) {
    if (!params || !params->is_object()) return json_null();
    auto id = params->get_string("id");
    if (!id) return json_string("error: id required");
    auto result = sandbox_.sandbox_json(*id);
    if (!result || result->is_null()) return json_string("not found");
    return result;
}

JsonPtr MCPServer::tool_sandbox_list(const JsonPtr& /*params*/) {
    return sandbox_.all_json();
}

JsonPtr MCPServer::tool_sandbox_violations(const JsonPtr& params) {
    size_t n = 0;
    if (params && params->is_object())
        if (auto num = params->get_number("last_n")) n = (size_t)*num;
    return sandbox_.violations_json(n);
}

// ── K-CUBE world state ──────────────────────────────────────────────────────────

JsonPtr MCPServer::tool_world_state(const JsonPtr& /*params*/) {
    world_.rebuild();
    return world_.to_json();
}

JsonPtr MCPServer::tool_field_dag(const JsonPtr& /*params*/) {
    world_.rebuild();
    world::FieldDag dag(world_, mu_reg_);
    dag.build();
    return json_string(dag.to_xml());
}

JsonPtr MCPServer::tool_field_dag_svg(const JsonPtr& /*params*/) {
    world_.rebuild();
    world::FieldDag dag(world_, mu_reg_);
    dag.build();
    return json_string(dag.to_svg());
}

} // namespace khanary::io
