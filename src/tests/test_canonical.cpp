#include "conformance.h"
#include "../util/canonical_hash.h"
#include <limits>
#include <fstream>
#include <iterator>

using namespace khanary;
int main(int argc, char** argv) {
    return test::run("canonical", [&](test::Suite& suite) {
        if (argc != 2) throw std::runtime_error("usage: CANONICAL_TEST canonical.json");
        std::ifstream file(argv[1], std::ios::binary);
        if (!file) throw std::runtime_error("missing canonical vectors");
        auto vectors = util::parse(std::string((std::istreambuf_iterator<char>(file)), {}));
        for (const auto& row : vectors->as_array()) {
            const auto id = row->get_string("id").value();
            auto input = util::parse(row->get_string("input").value());
            suite.check(id + "/bytes", util::stringify(input) == row->get_string("canonical").value());
            suite.check(id + "/hash", util::canonical_hash(*input) == row->get_string("sha256").value());
        }
        auto a = util::parse(R"({"z":-0.0,"a":1.0,"n":1.2345678901234567})");
        auto b = util::parse(R"({"n":1.2345678901234567,"a":1e0,"z":0})");
        suite.check("key_and_number_equivalence", util::canonical_hash(*a) == util::canonical_hash(*b));
        suite.check("precision_roundtrip", util::parse(util::stringify(a))->get_number("n") == a->get_number("n"));
        suite.check("empty_object_sha256", util::canonical_hash(*util::parse("{}")) ==
            "sha256:44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a");
        suite.check("unicode_escape_equivalence", util::stringify(util::parse("\"\\ud83d\\ude00\"")) == "\"\xf0\x9f\x98\x80\"");
        suite.check("control_and_key_roundtrip", util::stringify(util::parse(R"({"\"\u0001":"\b\f"})")) == R"({"\"\u0001":"\b\f"})");
        suite.check("exponent_normalization", util::stringify(util::json_number(1e-7)) == "1e-7");
        suite.rejects("non_finite", [] { (void)util::stringify(util::json_number(std::numeric_limits<double>::infinity())); });
        util::JsonObject fields;
        fields["topology_version"] = util::json_number(1);
        for (const char* key : {"topology_hash", "candidate_delta_hash", "certificate_hash"})
            fields[key] = util::json_string("sha256:" + std::string(64, '0'));
        auto identity = util::json_object(fields);
        const auto hash = util::cycle_identity_hash(*identity);
        identity->mut_object()["timestamp"] = util::json_number(999);
        suite.check("run_metadata_excluded", hash == util::cycle_identity_hash(*identity));
        identity->mut_object()["topology_version"] = util::json_number(2);
        suite.check("version_bound", hash != util::cycle_identity_hash(*identity));
        identity->mut_object()["certificate_hash"] = util::json_string("bad");
        suite.rejects("malformed_hash", [&] { (void)util::cycle_identity_hash(*identity); });
        suite.rejects("missing_identity", [] { (void)util::cycle_identity_hash(*util::json_object()); });
    });
}
