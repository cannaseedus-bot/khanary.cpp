#include "conformance.h"
#include "../util/schema.h"
#include "../manifest/manifest_registry.h"
#include "../micronaut/track_loader.h"
#include "../kuhul/xcfe_executor.h"
#include <fstream>
#include <iterator>

using namespace khanary;
namespace fs = std::filesystem;
static util::JsonPtr read(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("missing fixture " + path.filename().string());
    return util::parse(std::string((std::istreambuf_iterator<char>(file)), {}));
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
