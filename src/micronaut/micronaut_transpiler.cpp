#include "micronaut_transpiler.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <unordered_set>

namespace khanary::micronaut {

namespace {

std::string lower_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::vector<std::string> dedupe_copy(const std::vector<std::string>& values,
                                     bool lowercase) {
    std::vector<std::string> out;
    std::unordered_set<std::string> seen;
    for (auto value : values) {
        if (value.empty()) continue;
        if (lowercase) value = lower_copy(std::move(value));
        const auto marker = lower_copy(value);
        if (!seen.insert(marker).second) continue;
        out.push_back(std::move(value));
    }
    return out;
}

util::JsonArray strings_json(const std::vector<std::string>& values) {
    util::JsonArray out;
    out.reserve(values.size());
    for (const auto& value : values) out.push_back(util::json_string(value));
    return out;
}

std::string join_csv(const std::vector<std::string>& values) {
    std::string out;
    for (const auto& value : values) {
        if (!out.empty()) out.push_back(',');
        out += value;
    }
    return out.empty() ? "none" : out;
}

std::string safe_word(std::string id) {
    id = lower_copy(std::move(id));
    std::string out;
    out.reserve(id.size());
    bool dash = false;
    for (unsigned char c : id) {
        if (std::isalnum(c) || c == '_') {
            out.push_back(static_cast<char>(c));
            dash = false;
        } else if (!dash && !out.empty()) {
            out.push_back('-');
            dash = true;
        }
    }
    while (!out.empty() && out.back() == '-') out.pop_back();
    return out.empty() ? "micronaut" : out;
}

util::JsonObject contract_json(const RouteContract& contract) {
    util::JsonObject out;
    out["schemes"] = util::json_array(strings_json(dedupe_copy(contract.schemes, true)));
    out["verbs"] = util::json_array(strings_json(dedupe_copy(contract.verbs, true)));
    out["aiml_patterns"] = util::json_array(strings_json(dedupe_copy(contract.aiml_patterns, false)));
    out["aiml_templates"] = util::json_array(strings_json(dedupe_copy(contract.aiml_templates, false)));
    out["key_patterns"] = util::json_array(strings_json(dedupe_copy(contract.key_patterns, false)));
    out["key_templates"] = util::json_array(strings_json(dedupe_copy(contract.key_templates, false)));
    out["key_grammar"] = util::json_array(strings_json(dedupe_copy(contract.key_grammar, false)));
    out["tokens"] = util::json_array(strings_json(dedupe_copy(contract.tokens, true)));
    return out;
}

util::JsonObject class_projection_json(const MicronautEntry& entry,
                                       const RouteContract& contract) {
    util::JsonObject identity;
    identity["name"] = util::json_string(entry.id);
    identity["role"] = util::json_string(entry.description.empty() ? entry.category : entry.description);
    identity["version"] = util::json_null();
    identity["lineage"] = util::json_array();
    identity["birth_time"] = util::json_null();

    util::JsonObject context_window;
    context_window["size"] = util::json_number(4096.0);
    context_window["attention_focus"] = util::json_number(0.0);

    util::JsonObject temperature;
    temperature["current"] = util::json_number(0.7);
    temperature["min"] = util::json_number(0.0);
    temperature["max"] = util::json_number(2.0);
    temperature["adaptive"] = util::json_bool(true);

    util::JsonObject flare;
    flare["enthusiasm"] = util::json_number(0.5);
    flare["certainty"] = util::json_number(0.5);
    flare["verbosity"] = util::json_number(0.5);
    flare["empathy"] = util::json_number(0.5);

    util::JsonObject learning;
    learning["learning_rate"] = util::json_number(0.01);
    learning["curiosity"] = util::json_number(0.5);
    learning["forgetting_curve"] = util::json_string("exponential");
    learning["consolidation_frequency"] = util::json_number(3600.0);

    util::JsonObject example_defaults;
    example_defaults["context_window"] = util::json_object(std::move(context_window));
    example_defaults["temperature"] = util::json_object(std::move(temperature));
    example_defaults["response_flare"] = util::json_object(std::move(flare));
    example_defaults["learning_parameters"] = util::json_object(std::move(learning));

    util::JsonObject cognitive;
    cognitive["declared"] = util::json_object();
    cognitive["example_defaults"] = util::json_object(std::move(example_defaults));

    util::JsonObject memory_types;
    memory_types["working"] = util::json_string("active context items");
    memory_types["episodic"] = util::json_string("timestamped experiences");
    memory_types["semantic"] = util::json_string("knowledge graph concepts");
    memory_types["procedural"] = util::json_string("learned execution procedures");

    util::JsonObject memory;
    memory["declared_patterns"] = util::json_array();
    memory["jrom_role"] = util::json_string("provenance_only");
    memory["example_types"] = util::json_object(std::move(memory_types));

    util::JsonArray fold_ops;
    for (const auto& fold : entry.folds)
        if (!fold.op.empty()) fold_ops.push_back(util::json_string(fold.op));

    util::JsonObject execution;
    execution["primary_phase"] = util::json_string(kuhul::phase_name(entry.primary_phase));
    execution["fold_ops"] = util::json_array(std::move(fold_ops));
    execution["schemes"] = util::json_array(strings_json(dedupe_copy(contract.schemes, true)));
    execution["verbs"] = util::json_array(strings_json(dedupe_copy(contract.verbs, true)));
    execution["scheduler"] = util::json_string("phase_gated_runtime");

    util::JsonObject methods;
    methods["execute"] = util::json_string("host_handler_or_executor; never implied by this projection");
    methods["learn"] = util::json_string("Factory/Xul-gated proposal path only");
    methods["update_context"] = util::json_string("runtime attention/JROM observation path");
    methods["adjust_temperature"] = util::json_string("model profile concern, not routing authority");
    methods["modulate_flare"] = util::json_string("style projection concern, not fact authority");
    methods["remember"] = util::json_string("memory surfaces are proposal/recall evidence");
    methods["forget"] = util::json_string("requires explicit policy and authority");
    methods["communicate"] = util::json_string("typed envelopes over RPC/MCP with allowlists");

    util::JsonObject projection;
    projection["schema"] = util::json_string("micronaut.class_projection.example.v1");
    projection["example_only"] = util::json_bool(true);
    projection["authority_boundary"] = util::json_string(
        "example_projection_not_law; runtime authority remains track schema + XCFE + Ch'en + Xul");
    projection["identity"] = util::json_object(std::move(identity));
    projection["cognitive_state"] = util::json_object(std::move(cognitive));
    projection["memory_system"] = util::json_object(std::move(memory));
    projection["execution_engine"] = util::json_object(std::move(execution));
    projection["capabilities"] = util::json_array(strings_json(dedupe_copy(entry.capabilities, false)));
    projection["methods"] = util::json_object(std::move(methods));
    return projection;
}

} // namespace

util::JsonPtr MicronautTranspiler::compile_entry(const MicronautEntry& entry) {
    const auto schemes = dedupe_copy(entry.route_contract.schemes, true);
    const auto verbs = dedupe_copy(entry.route_contract.verbs, true);
    const auto tokens = dedupe_copy(entry.route_contract.tokens, true);
    const auto aiml_patterns = dedupe_copy(entry.route_contract.aiml_patterns, false);
    const auto aiml_templates = dedupe_copy(entry.route_contract.aiml_templates, false);
    const auto key_patterns = dedupe_copy(entry.route_contract.key_patterns, false);
    const auto key_templates = dedupe_copy(entry.route_contract.key_templates, false);
    const auto key_grammar = dedupe_copy(entry.route_contract.key_grammar, false);

    std::ostringstream definition;
    definition << entry.id << " callable route contract: "
               << "schemes=" << join_csv(schemes)
               << "; verbs=" << join_csv(verbs)
               << "; patterns=" << (aiml_patterns.size() + key_patterns.size())
               << "; templates=" << (aiml_templates.size() + key_templates.size())
               << "; key_grammar=" << key_grammar.size()
               << "; tokens=" << join_csv(tokens);

    const std::string word = safe_word(entry.id);
    util::JsonObject row;
    row["schema"] = util::json_string("key.grammar.micronaut_route.v1");
    row["word"] = util::json_string(word + "-route");
    row["key"] = util::json_string("MICRONAUT:" + lower_copy(word) + ":ROUTE");
    row["unit"] = util::json_string("route_contract");
    row["class"] = util::json_string("MICRONAUT_ROUTE");
    row["book"] = util::json_string("Book(" + lower_copy(word) + ")");
    row["definition"] = util::json_string(definition.str());
    row["lexicon_source"] = util::json_string("runtime.micronaut_registry");
    row["track"] = util::json_string(entry.id);
    row["phase"] = util::json_string(kuhul::phase_name(entry.primary_phase));
    row["route_contract"] = util::json_object(contract_json(entry.route_contract));
    row["micronaut_class"] = util::json_object(
        class_projection_json(entry, entry.route_contract));
    if (entry.profile_projection)
        row["micronaut_profile"] = entry.profile_projection;
    row["fact_authority"] = util::json_bool(false);
    row["authority_boundary"] = util::json_string(
        "route_contract_declares_callable_surface_not_fact");
    row["transpiler_side"] = util::json_string("cpp_runtime_projection");
    return util::json_object(std::move(row));
}

util::JsonArray MicronautTranspiler::compile_registry(
    const MicronautRegistry& registry,
    MicronautTranspileOptions options)
{
    util::JsonArray rows;
    auto ids = registry.ids();
    std::sort(ids.begin(), ids.end());
    for (const auto& id : ids) {
        const auto* entry = registry.get(id);
        if (!entry) continue;
        if (!options.include_inactive && !entry->active) continue;
        rows.push_back(compile_entry(*entry));
    }
    return rows;
}

} // namespace khanary::micronaut
