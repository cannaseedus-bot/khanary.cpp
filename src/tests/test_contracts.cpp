#include "conformance.h"
#include "../util/schema.h"
#include "../manifest/manifest_registry.h"
#include "../micronaut/track_loader.h"
#include "../kuhul/xcfe_executor.h"
#include <algorithm>
#include <fstream>
#include <iterator>
#include <vector>

using namespace khanary;
namespace fs = std::filesystem;
static util::JsonPtr read(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("missing fixture " + path.filename().string());
    return util::parse(std::string((std::istreambuf_iterator<char>(file)), {}));
}
static bool has_string(const std::vector<std::string>& values, const std::string& target) {
    return std::find(values.begin(), values.end(), target) != values.end();
}
int main(int argc, char** argv) {
    return test::run("contracts", [&](test::Suite& suite) {
        if (argc != 3) throw std::runtime_error("usage: CONTRACT_TEST vectors.json data-dir");
        const fs::path data = argv[2];
        const fs::path scratch = fs::current_path() / "contract-fixtures";
        fs::create_directories(scratch / "schema");
        fs::create_directories(scratch / "tracks");
        fs::create_directories(scratch / "manifest");
        for (const auto& file : fs::directory_iterator(data / "schema"))
            if (file.path().extension() == ".json")
                fs::copy_file(file.path(), scratch / "schema" / file.path().filename(), fs::copy_options::overwrite_existing);
        auto vectors = read(argv[1]);
        for (const auto& row : vectors->as_array()) {
            const auto id = row->get_string("id").value();
            const auto kind = row->get_string("kind").value();
            const bool expected = row->get_bool("valid").value();
            if (kind == "json") {
                bool accepted = true;
                try { (void)util::parse(row->get_string("text").value()); }
                catch (const std::exception&) { accepted = false; }
                suite.check(id, accepted == expected);
                continue;
            }
            if (kind != "track" && kind != "manifest") throw std::runtime_error("unknown fixture kind");
            const auto* value = row->get("value");
            if (!value) throw std::runtime_error("missing fixture value");
            const auto schema = data / "schema" / (kind == "track" ? "micronaut-track.schema.json" : "manifest-map.schema.json");
            suite.check(id + "/schema", util::validate_schema_file(*value, schema) == expected);
            const fs::path path = scratch / (kind == "track" ? "tracks/probe.semantic-tracks.v1.json" : "manifest/probe.manifest.json");
            { std::ofstream f(path, std::ios::binary); f << util::stringify(*value); }
            bool accepted;
            if (kind == "track") {
                micronaut::MicronautRegistry registry;
                micronaut::TrackLoader loader(registry);
                accepted = loader.load_file(path.string());
                suite.check(id + "/atomic", accepted || registry.ids().empty());
            } else {
                manifest::ManifestRegistry registry;
                accepted = true;
                try { registry.load(path.string()); } catch (const std::exception&) { accepted = false; }
                suite.check(id + "/atomic", accepted || registry.size() == 0);
                if (accepted && id == "manifest_conditional") {
                    kuhul::XCFEExecutor xcfe;
                    const auto* tokens = registry.raw("a");
                    const auto node = util::parse(tokens->at(0));
                    suite.check("conditional/then", util::stringify(xcfe.eval(node, {{"user.authenticated", "true"}})) == "[\"yes\"]");
                    suite.check("conditional/else", util::stringify(xcfe.eval(node, {{"user.authenticated", "false"}})) == "[\"no\"]");
                }
            }
            suite.check(id + "/loader", accepted == expected);
        }

        {
            util::JsonObject route_contract;
            route_contract["schemes"] = util::json_array({util::json_string("DECISION"), util::json_string("decision")});
            route_contract["verbs"] = util::json_array({util::json_string("Choose"), util::json_string("choose")});
            route_contract["aiml_patterns"] = util::json_array({util::json_string("DECISION *")});
            route_contract["aiml_templates"] = util::json_array({util::json_string("decision://choose")});
            route_contract["key_patterns"] = util::json_array({
                util::json_string("decision:choose:*"), util::json_string("decision:choose:*")});
            route_contract["key_templates"] = util::json_array({util::json_string("decision:result:selected")});
            route_contract["key_grammar"] = util::json_array({util::json_string("schema:scfml.pattern_learning.v1")});

            util::JsonObject track;
            track["@no-eval"] = util::json_bool(true);
            track["id"] = util::json_string("ROUTE-CONTRACT-TEST-µ");
            track["phase"] = util::json_string("Sek");
            track["governs"] = util::json_array({util::json_string("decision")});
            track["route_contract"] = util::json_object(std::move(route_contract));
            track["tokens"] = util::json_array({util::json_string("Action"), util::json_string("action"), util::json_string("")});
            util::JsonObject brain;
            brain["template"] = util::json_string("analyst");
            brain["projection_only"] = util::json_bool(true);
            brain["authority_boundary"] = util::json_string("example_only");
            track["brain_profile"] = util::json_object(std::move(brain));
            util::JsonObject progression;
            progression["level"] = util::json_number(5.0);
            progression["projection_only"] = util::json_bool(true);
            track["progression"] = util::json_object(std::move(progression));

            const auto schema = data / "schema" / "micronaut-track.schema.json";
            const auto track_value = util::json_object(std::move(track));
            suite.check("route_contract_key_grammar_schema",
                        util::validate_schema_file(*track_value, schema));

            const fs::path path = scratch / "tracks/route-contract-test.semantic-tracks.v1.json";
            { std::ofstream f(path, std::ios::binary); f << util::stringify(*track_value); }
            micronaut::MicronautRegistry registry;
            micronaut::TrackLoader track_loader(registry);
            const bool loaded = track_loader.load_file(path.string());
            suite.check("route_contract_key_grammar_loader", loaded);
            const auto* entry = registry.get("ROUTE-CONTRACT-TEST-µ");
            suite.check("route_contract_key_grammar_entry", entry != nullptr);
            if (entry) {
                suite.check("route_contract_schemes_normalized",
                            has_string(entry->route_contract.schemes, "decision") &&
                            entry->route_contract.schemes.size() == 1);
                suite.check("route_contract_verbs_normalized",
                            has_string(entry->route_contract.verbs, "choose") &&
                            entry->route_contract.verbs.size() == 1);
                suite.check("route_contract_key_patterns_preserved",
                            has_string(entry->route_contract.key_patterns, "decision:choose:*") &&
                            entry->route_contract.key_patterns.size() == 1);
                suite.check("route_contract_key_templates_preserved",
                            has_string(entry->route_contract.key_templates, "decision:result:selected"));
                suite.check("route_contract_key_grammar_preserved",
                            has_string(entry->route_contract.key_grammar, "schema:scfml.pattern_learning.v1"));
                suite.check("route_contract_tokens_normalized",
                            has_string(entry->route_contract.tokens, "action") &&
                            entry->route_contract.tokens.size() == 1);
                suite.check("profile_projection_loaded",
                            entry->profile_projection && entry->profile_projection->is_object());
                if (entry->profile_projection) {
                    suite.check("profile_projection_boundary",
                                entry->profile_projection->get_bool("projection_only").value_or(false) &&
                                entry->profile_projection->get("brain_profile") != nullptr &&
                                entry->profile_projection->get("progression") != nullptr);
                }
            }
        }

        micronaut::MicronautRegistry tracks;
        micronaut::TrackLoader loader(tracks);
        for (const auto& entry : fs::directory_iterator(data / "tracks"))
            if (entry.path().extension() == ".json")
                suite.check("shipped/" + entry.path().filename().string(), loader.load_file(entry.path().string()));
        for (const auto& entry : fs::directory_iterator(data / "manifest")) {
            if (entry.path().extension() != ".json") continue;
            manifest::ManifestRegistry registry;
            bool ok = true;
            try { registry.load(entry.path().string()); } catch (const std::exception&) { ok = false; }
            suite.check("shipped/" + entry.path().filename().string(), ok);
        }
        suite.rejects("invalid_utf8", [] { (void)util::parse(std::string("\"\xc0\xaf\"")); });
        suite.rejects("nesting_limit", [] { (void)util::parse(std::string(257, '[') + "0" + std::string(257, ']')); });
    });
}
