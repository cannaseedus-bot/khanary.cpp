#include "conformance.h"
// Acceptance tests for project_v().
//
// Test 1 (determinism):  Two traces with identical authorized events but different
//   absolute seq numbers → byte-identical text and projection_hash.
//   Validates that within-dim seq-sort + verbatim payload_json produce stable output.
//
// Test 2 (r≈0):  Pass the same µ-track scores as both s_mu_projection and s_mu_dds
//   into Gyroscope::observe().  Δµ = 0 → ΔABI = 0 → r = 0 exactly.
//   First end-to-end run of the full project_v() → score → Gyroscope pipeline.

#include "../io/projection.h"
#include "../world/gyroscope.h"
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

using namespace khanary;

static void test_determinism(const char* proj_book) {
    io::ProjectionEngine engine(proj_book);

    // Trace A and B: same event types, same payloads, different absolute seq values.
    io::JROMTrace trace_A = {
        {.type="decompose_task", .payload_json="{\"task\":\"analyze X\"}", .seq=1},
        {.type="score_evidence", .payload_json="{\"verdict\":0.7}",        .seq=2},
        {.type="verify_result",  .payload_json="{\"result\":\"passed\"}",  .seq=3},
    };
    io::JROMTrace trace_B = {
        {.type="decompose_task", .payload_json="{\"task\":\"analyze X\"}", .seq=100},
        {.type="score_evidence", .payload_json="{\"verdict\":0.7}",        .seq=200},
        {.type="verify_result",  .payload_json="{\"result\":\"passed\"}",  .seq=300},
    };

    io::ProjectionRequest req;
    // abi_version, target, thinking use their defaults

    auto p_A = engine.project_v(trace_A, req);
    auto p_B = engine.project_v(trace_B, req);

    if (!p_A.ok) { std::cerr << "[FAIL] determinism: p_A error: " << p_A.error << "\n"; throw khanary::test::Failure("projection assertion"); }
    if (!p_B.ok) { std::cerr << "[FAIL] determinism: p_B error: " << p_B.error << "\n"; throw khanary::test::Failure("projection assertion"); }
    if (p_A.projection_hash != p_B.projection_hash) {
        std::cerr << "[FAIL] determinism: hashes differ\n"
                  << "  A(" << p_A.projection_hash << "):\n" << p_A.text
                  << "\n  B(" << p_B.projection_hash << "):\n" << p_B.text << "\n";
        throw khanary::test::Failure("projection assertion");
    }
    std::cout << "[PASS] determinism  hash=" << p_A.projection_hash << "\n";
}

static void test_r_zero(const char* proj_book, const char* gyro_book) {
    io::ProjectionEngine proj_engine(proj_book);
    world::Gyroscope     gyro(gyro_book);

    io::JROMTrace trace = {
        {.type="decompose_task", .payload_json="{\"task\":\"plan\"}", .seq=1},
        {.type="verify_result",  .payload_json="{\"result\":\"ok\"}", .seq=2},
    };

    io::ProjectionRequest req;
    auto p = proj_engine.project_v(trace, req);
    if (!p.ok) { std::cerr << "[FAIL] r≈0: project_v error: " << p.error << "\n"; throw khanary::test::Failure("projection assertion"); }

    // Tokenize the projection text (whitespace split)
    std::vector<std::string> tokens;
    std::istringstream iss(p.text);
    for (std::string tok; iss >> tok;) tokens.push_back(tok);

    // Each µ-track governs exactly the projection's tokens → coverage = 1.0 for every track.
    // Using identical scores on both sides guarantees Δµ = 0 → r = 0 exactly.
    std::array<std::unordered_set<std::string>, world::TRACK_DIM_COUNT> governs;
    for (auto& s : governs)
        for (const auto& t : tokens) s.insert(t);

    auto scores = world::Gyroscope::score_tokens(tokens, governs);
    world::GyroInputs inputs{ .omega=1.0f, .m_dds=0.5f, .k=2.0f };
    auto result = gyro.observe(scores, scores, inputs);

    if (std::fabs(result.r) > 1e-5f) {
        std::cerr << "[FAIL] r≈0 test: r=" << result.r
                  << " (expected < 1e-5)\n";
        throw khanary::test::Failure("projection assertion");
    }
    std::cout << "[PASS] r≈0          r=" << result.r
              << " gamma=" << result.gamma
              << " stable=" << result.dynamically_stable
              << " dominant=" << result.dominant_anchor << "\n";
}

static int acceptance_main(int argc, char** argv) {
    const char* proj_book = "../../data/projection_abi_v1.json";
    const char* gyro_book = "../../data/gyro_abi_v1.json";
    if (argc >= 2) proj_book = argv[1];
    if (argc >= 3) gyro_book = argv[2];

    test_determinism(proj_book);
    test_r_zero(proj_book, gyro_book);
    std::cout << "All projection tests passed.\n";
    return 0;
}

int main(int argc, char** argv) { return khanary::test::legacy("projection", [&] { return acceptance_main(argc, argv); }); }
