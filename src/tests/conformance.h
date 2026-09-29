#pragma once
#include "../util/json.h"
#include "../util/sha256.h"
#include <algorithm>
#include <functional>
#include <iostream>
#include <sstream>

namespace khanary::test {
// Build/conformance identity. A verdict is not evidence unless it identifies
// which executable produced it and which conformance set it contained:
// SourcePresence != ExecutedArtifact. build_stamp is this translation unit's
// compile time, so a stale executable is distinguishable from a current one
// even when the case count happens to match.
inline constexpr const char* kConformanceVersion = "1";
inline const std::string& build_stamp() {
    static const std::string stamp = std::string(__DATE__) + " " + __TIME__;
    return stamp;
}
struct Failure : std::runtime_error { using std::runtime_error::runtime_error; };
struct Suite {
    std::vector<std::pair<std::string, bool>> cases;
    void check(const std::string& name, bool passed) { cases.emplace_back(name, passed); }
    template<class F> void rejects(const std::string& name, F operation) {
        bool rejected = false;
        try { operation(); } catch (const std::exception&) { rejected = true; }
        check(name, rejected);
    }
};
inline int run(const std::string& name, const std::function<void(Suite&)>& body,
               size_t expected_count = 0) {
    Suite suite;
    std::string error;
    try { body(suite); }
    catch (const Failure&) { suite.check("assertion", false); }
    catch (const std::exception& e) { error = e.what(); }
    std::sort(suite.cases.begin(), suite.cases.end());
    util::JsonArray rows;
    std::string seeds;
    size_t failures = 0;
    for (const auto& [id, passed] : suite.cases) {
        if (!seeds.empty()) seeds += '\n';
        seeds += name + "|" + id + "|" + (passed ? "PASS" : "FAIL");
        if (!passed) ++failures;
        rows.push_back(util::json_object({{"id", util::json_string(id)},
            {"verdict", util::json_string(passed ? "PASS" : "FAIL")}}));
    }
    const bool count_ok = expected_count == 0 || suite.cases.size() == expected_count;
    const int code = !error.empty() || suite.cases.empty() ? 2
                   : !count_ok ? 2
                   : failures ? 1 : 0;
    auto result = util::json_object({
        {"@kind", util::json_string("khanary.contract.verdict.v1")},
        {"suite", util::json_string(name)}, {"cases", util::json_array(std::move(rows))},
        {"conformance_version", util::json_string(kConformanceVersion)},
        {"build_stamp", util::json_string(build_stamp())},
        {"case_count", util::json_number(static_cast<double>(suite.cases.size()))},
        {"verdict", util::json_string(code == 0 ? "PASS" : code == 1 ? "FAIL" : "ERROR")},
        {"exit_code", util::json_number(code)},
        {"summary_hash", util::json_string("sha256:" + util::sha256_hex(seeds))}
    });
    if (expected_count > 0) {
        result->mut_object()["expected_count"] = util::json_number(static_cast<double>(expected_count));
        if (!count_ok)
            result->mut_object()["count_mismatch"] = util::json_string(
                "expected " + std::to_string(expected_count) +
                " got " + std::to_string(suite.cases.size()));
    }
    if (!error.empty()) result->mut_object()["error"] = util::json_string(error);
    std::cout << util::stringify(result) << '\n';
    return code;
}
inline int legacy(const std::string& name, const std::function<int()>& body) {
    return run(name, [&](Suite& suite) {
        std::ostringstream human;
        struct Restore {
            std::streambuf* old;
            ~Restore() { std::cout.rdbuf(old); }
        };
        int code;
        {
            Restore restore{std::cout.rdbuf(human.rdbuf())};
            code = body();
        }
        std::cerr << human.str();
        suite.check("acceptance", code == 0);
    });
}
}
