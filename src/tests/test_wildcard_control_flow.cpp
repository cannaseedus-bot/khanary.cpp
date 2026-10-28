//
// KHANARY_WILDCARD_CONTROL_FLOW_TEST — Q-Prime questions over wildcard control
// flow.
//
// The three productions are the point: THEN, ELSE and BUT are not interchangeable.
// ELSE without a known THEN is a QUERY_GAP rather than an invented branch, and BUT
// is a contrast rather than an ELSE. A match is not a resolved condition, and a
// resolved condition is not authorized execution — so nothing here executes.
//
// The legality question is wired to the real KFA gate, so a proposed branch that
// does not compose is refused with the KFA code instead of being asked about.
//
// Usage: KHANARY_WILDCARD_CONTROL_FLOW_TEST <data-root>   (default: "data")
//
#include "conformance.h"
#include "../kuhul/fold_algebra.h"
#include "../kuhul/fold_gate.h"
#include "../micronaut/wildcard_control_flow.h"

#include <fstream>
#include <sstream>
#include <string>

namespace {

using khanary::micronaut::AnalysisOptions;
using khanary::micronaut::ConditionOperator;
using khanary::micronaut::ContinuationMarker;
using khanary::micronaut::ControlFlowAnalysis;
using khanary::micronaut::ControlFlowPattern;
using khanary::micronaut::GapRoute;
using khanary::micronaut::OperandKind;
using khanary::micronaut::QuestionDimension;

std::string read_file(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open " + path);
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}

bool has_dimension(const ControlFlowAnalysis& analysis, QuestionDimension dimension) {
    for (const auto& question : analysis.questions) {
        if (question.dimension == dimension) return true;
    }
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string data_root = argc > 1 ? argv[1] : "data";

    return khanary::test::run("wildcard_control_flow",
                              [&](khanary::test::Suite& suite) {
        namespace kfa = khanary::kuhul::foldalgebra;

        // ── 1. The three productions ───────────────────────────────────────
        const auto then_case = khanary::micronaut::parse_control_flow_pattern(
            "* IF * OR * THEN * AND *");
        const auto else_case = khanary::micronaut::parse_control_flow_pattern(
            "* IF * OR * ELSE * AND *");
        const auto but_case = khanary::micronaut::parse_control_flow_pattern(
            "* IF * OR * BUT * AND *");

        suite.check("parse/then-ok", then_case.ok);
        suite.check("parse/else-ok", else_case.ok);
        suite.check("parse/but-ok", but_case.ok);

        suite.check("parse/then-marker", then_case.pattern.marker == ContinuationMarker::Then);
        suite.check("parse/else-marker", else_case.pattern.marker == ContinuationMarker::Else);
        suite.check("parse/but-marker", but_case.pattern.marker == ContinuationMarker::But);

        suite.check("parse/condition-op-is-or",
                    then_case.pattern.condition_op == ConditionOperator::Or);
        suite.check("parse/five-captures", then_case.pattern.capture_count == std::size_t{5});
        suite.check("parse/continuation-op", then_case.pattern.continuation_op == "AND");
        suite.check("parse/leading-capture", then_case.pattern.leading.size() == std::size_t{1});

        // Slots are numbered in order of appearance, 1-based.
        suite.check("parse/slot-order",
                    then_case.pattern.leading[0].slot == 1 &&
                        then_case.pattern.condition_operands[0].slot == 2 &&
                        then_case.pattern.condition_operands[1].slot == 3 &&
                        then_case.pattern.continuation_operands[0].slot == 4 &&
                        then_case.pattern.continuation_operands[1].slot == 5);

        // A conjunctive condition keeps its operator.
        const auto and_condition = khanary::micronaut::parse_control_flow_pattern(
            "* IF * AND * THEN * AND *");
        suite.check("parse/and-condition-preserved",
                    and_condition.ok &&
                        and_condition.pattern.condition_op == ConditionOperator::And);

        // ── 2. ELSE without a THEN is a question, not a branch ─────────────
        {
            const ControlFlowAnalysis unknown =
                khanary::micronaut::analyze_control_flow("* IF * OR * ELSE * AND *");
            suite.check("else/without-then-is-a-gap",
                        unknown.blocking_gap == GapRoute::QueryGap);
            suite.check("else/without-then-detail",
                        unknown.blocking_detail == "missing_branch");
            suite.check("else/question-is-asked", has_dimension(unknown, QuestionDimension::Continuation));
            suite.check("else/never-executes", !unknown.executes);

            AnalysisOptions known;
            known.then_branch_known = true;
            const ControlFlowAnalysis resolved =
                khanary::micronaut::analyze_control_flow("* IF * OR * ELSE * AND *", known);
            suite.check("else/with-then-is-not-a-gap", resolved.blocking_gap == GapRoute::None);
        }

        // ── 3. BUT is a contrast, never an alias for ELSE ──────────────────
        {
            AnalysisOptions known;
            known.then_branch_known = true;
            const ControlFlowAnalysis contrast =
                khanary::micronaut::analyze_control_flow("* IF * OR * BUT * AND *", known);
            suite.check("but/is-a-gap", contrast.blocking_gap == GapRoute::IntentGap);
            suite.check("but/gap-detail",
                        contrast.blocking_detail.find("BUT") != std::string::npos);
            // Even with a THEN branch known, BUT does not become an alternative.
            suite.check("but/not-converted-to-else",
                        contrast.parsed.pattern.marker == ContinuationMarker::But);
            suite.check("but/question-asked",
                        has_dimension(contrast, QuestionDimension::Continuation));
        }

        // ── 4. SRAI is its own AST node ────────────────────────────────────
        {
            const auto with_target = khanary::micronaut::parse_control_flow_pattern(
                "* IF * OR <srai>candidate B</srai> THEN * AND *");
            suite.check("srai/parses", with_target.ok);
            suite.check("srai/is-its-own-node",
                        with_target.pattern.condition_operands.size() == std::size_t{2} &&
                            with_target.pattern.condition_operands[1].kind ==
                                OperandKind::SraiCall);
            suite.check("srai/target-recorded",
                        with_target.pattern.condition_operands[1].srai_target == "candidate B");
            suite.check("srai/does-not-consume-a-capture-slot",
                        with_target.pattern.capture_count == std::size_t{4});
            suite.check("srai/flagged", with_target.pattern.has_srai);

            const ControlFlowAnalysis resolved =
                khanary::micronaut::analyze_control_flow(
                    "* IF * OR <srai>candidate B</srai> THEN * AND *");
            suite.check("srai/target-question-asked",
                        has_dimension(resolved, QuestionDimension::Srai));
            suite.check("srai/resolved-target-is-not-a-gap",
                        resolved.blocking_gap == GapRoute::None);

            const ControlFlowAnalysis unresolved =
                khanary::micronaut::analyze_control_flow("* IF * OR <srai/> THEN * AND *");
            suite.check("srai/self-closing-is-knowledge-gap",
                        unresolved.blocking_gap == GapRoute::KnowledgeGap);
            suite.check("srai/self-closing-detail",
                        unresolved.blocking_detail == "unknown_srai_target");
        }

        // ── 5. Parse errors ────────────────────────────────────────────────
        {
            const auto missing_if =
                khanary::micronaut::parse_control_flow_pattern("* THEN *");
            suite.check("errors/missing-if",
                        !missing_if.ok && missing_if.error_code == "MISSING_IF");

            const auto missing_op =
                khanary::micronaut::parse_control_flow_pattern("* IF * * THEN * AND *");
            suite.check("errors/missing-operator",
                        !missing_op.ok && missing_op.error_code == "MISSING_OPERATOR");

            const auto missing_marker =
                khanary::micronaut::parse_control_flow_pattern("* IF * OR * * AND *");
            suite.check("errors/missing-marker",
                        !missing_marker.ok && missing_marker.error_code == "MISSING_MARKER");

            const auto unbalanced = khanary::micronaut::parse_control_flow_pattern(
                "* IF <srai>broken OR * THEN * AND *");
            suite.check("errors/unbalanced-srai",
                        !unbalanced.ok && unbalanced.error_code == "UNBALANCED_SRAI");

            const auto literal_lead = khanary::micronaut::parse_control_flow_pattern(
                "hello IF * OR * THEN * AND *");
            suite.check("errors/unexpected-leading-token",
                        !literal_lead.ok && literal_lead.error_code == "UNEXPECTED_TOKEN");

            const auto trailing = khanary::micronaut::parse_control_flow_pattern(
                "* IF * OR * THEN * AND * extra");
            suite.check("errors/trailing-token",
                        !trailing.ok && trailing.error_code == "UNEXPECTED_TOKEN");

            // A parse failure is itself a gap, not an exception.
            const ControlFlowAnalysis broken =
                khanary::micronaut::analyze_control_flow("* THEN *");
            suite.check("errors/parse-failure-is-a-query-gap",
                        broken.blocking_gap == GapRoute::QueryGap);
        }

        // ── 6. All six question dimensions are asked ───────────────────────
        {
            const ControlFlowAnalysis analysis =
                khanary::micronaut::analyze_control_flow("* IF * OR <srai>x</srai> THEN * AND *");
            suite.check("questions/capture", has_dimension(analysis, QuestionDimension::Capture));
            suite.check("questions/binding", has_dimension(analysis, QuestionDimension::Binding));
            suite.check("questions/condition", has_dimension(analysis, QuestionDimension::Condition));
            suite.check("questions/continuation",
                        has_dimension(analysis, QuestionDimension::Continuation));
            suite.check("questions/srai", has_dimension(analysis, QuestionDimension::Srai));
            suite.check("questions/legality", has_dimension(analysis, QuestionDimension::Legality));
            suite.check("questions/never-executes", !analysis.executes);
        }

        // ── 7. Legality is wired to the real KFA gate ──────────────────────
        {
            const kfa::FoldGate gate = kfa::FoldGate::load(data_root);
            suite.check("legality/gate-loaded", gate.active());

            AnalysisOptions options;
            options.legality = [&gate](const std::vector<std::string>& operators, std::string* code,
                                       std::string* detail) {
                const kfa::GateVerdict verdict = gate.admit(operators);
                if (code) *code = verdict.code;
                if (detail) *detail = verdict.detail;
                return verdict.admitted;
            };
            options.proposed_operators = {"SCORE", "TRAIN"};

            const ControlFlowAnalysis refused =
                khanary::micronaut::analyze_control_flow("* IF * OR * THEN * AND *", options);
            suite.check("legality/refused-branch-is-xcfe-reject",
                        refused.blocking_gap == GapRoute::XcfeReject);
            suite.check("legality/refusal-code-preserved",
                        refused.blocking_detail == "REQUIRES_VERIFICATION");

            AnalysisOptions admitted = options;
            admitted.proposed_operators = {"VERIFY", "TRAIN"};
            const ControlFlowAnalysis ok =
                khanary::micronaut::analyze_control_flow("* IF * OR * THEN * AND *", admitted);
            suite.check("legality/admitted-branch-is-not-a-gap",
                        ok.blocking_gap == GapRoute::None);

            // Precedence: a refused branch outranks an ambiguous marker.
            AnalysisOptions both = options;
            both.proposed_operators = {"SCORE", "TRAIN"};
            const ControlFlowAnalysis but_refused =
                khanary::micronaut::analyze_control_flow("* IF * OR * BUT * AND *", both);
            suite.check("legality/refusal-outranks-intent-gap",
                        but_refused.blocking_gap == GapRoute::XcfeReject);
        }

        // ── 8. XOR: the capture-difference mask ────────────────────────────
        {
            const auto a = khanary::micronaut::parse_control_flow_pattern(
                "* IF * OR <srai/> THEN * AND *");
            const auto b = khanary::micronaut::parse_control_flow_pattern(
                "* IF * OR * THEN * AND *");
            suite.check("xor/both-parse", a.ok && b.ok);

            const auto mask = khanary::micronaut::capture_difference(a.pattern, b.pattern, 5);
            suite.check("xor/one-slot-differs", khanary::graphscript::popcount(mask) == std::size_t{1});
            suite.check("xor/axis-is-the-displaced-slot",
                        khanary::graphscript::differing_axis(mask,
                                                             khanary::graphscript::BitVector::zeros(5)) == 4);

            const auto then_pattern = khanary::micronaut::parse_control_flow_pattern(
                "* IF * OR * THEN * AND *");
            const auto else_pattern = khanary::micronaut::parse_control_flow_pattern(
                "* IF * OR * ELSE * AND *");
            const auto same_arity = khanary::micronaut::capture_difference(
                then_pattern.pattern, else_pattern.pattern, 5);
            suite.check("xor/same-arity-differs-nowhere",
                        khanary::graphscript::popcount(same_arity) == std::size_t{0});
        }

        // ── 9. Fixtures: the same file the Python checker reads ────────────
        {
            auto document = khanary::util::parse(
                read_file(data_root + "/conformance/wildcard_control_flow.json"));

            for (const auto& entry : document->get("parse_ok")->as_array()) {
                if (entry == nullptr) continue;
                const std::string id = entry->get_string("id").value_or("?");
                AnalysisOptions options;
                options.then_branch_known = entry->get_bool("then_branch_known").value_or(false);
                const ControlFlowAnalysis analysis = khanary::micronaut::analyze_control_flow(
                    entry->get_string("pattern").value_or(""), options);
                const std::string expected_gap = entry->get_string("gap").value_or("NONE");
                const std::string marker = entry->get_string("marker").value_or("");
                const bool marker_ok =
                    khanary::micronaut::continuation_marker_name(analysis.parsed.pattern.marker) ==
                    marker;
                suite.check("fixtures/parse/" + id,
                            analysis.parsed.ok && marker_ok &&
                                khanary::micronaut::gap_route_name(analysis.blocking_gap) ==
                                    expected_gap);
            }

            for (const auto& entry : document->get("parse_errors")->as_array()) {
                if (entry == nullptr) continue;
                const std::string id = entry->get_string("id").value_or("?");
                const auto parsed = khanary::micronaut::parse_control_flow_pattern(
                    entry->get_string("pattern").value_or(""));
                suite.check("fixtures/error/" + id,
                            !parsed.ok &&
                                parsed.error_code == entry->get_string("error").value_or(""));
            }

            const kfa::FoldGate gate = kfa::FoldGate::load(data_root);
            for (const auto& entry : document->get("legality")->as_array()) {
                if (entry == nullptr) continue;
                const std::string id = entry->get_string("id").value_or("?");
                AnalysisOptions options;
                options.legality = [&gate](const std::vector<std::string>& operators,
                                           std::string* code, std::string* detail) {
                    const kfa::GateVerdict verdict = gate.admit(operators);
                    if (code) *code = verdict.code;
                    if (detail) *detail = verdict.detail;
                    return verdict.admitted;
                };
                for (const auto& name : entry->get("proposed_operators")->as_array()) {
                    if (name != nullptr && name->is_string()) {
                        options.proposed_operators.push_back(name->as_string());
                    }
                }
                const ControlFlowAnalysis analysis = khanary::micronaut::analyze_control_flow(
                    entry->get_string("pattern").value_or(""), options);
                suite.check("fixtures/legality/" + id,
                            khanary::micronaut::gap_route_name(analysis.blocking_gap) ==
                                entry->get_string("gap").value_or("NONE"));
            }

            for (const auto& entry : document->get("xor")->as_array()) {
                if (entry == nullptr) continue;
                const std::string id = entry->get_string("id").value_or("?");
                const auto a = khanary::micronaut::parse_control_flow_pattern(
                    entry->get_string("a").value_or(""));
                const auto b = khanary::micronaut::parse_control_flow_pattern(
                    entry->get_string("b").value_or(""));
                const std::size_t width =
                    static_cast<std::size_t>(entry->get_number("width").value_or(5.0));
                const std::size_t expected =
                    static_cast<std::size_t>(entry->get_number("expect_popcount").value_or(-1.0));
                suite.check("fixtures/xor/" + id,
                            a.ok && b.ok &&
                                khanary::graphscript::popcount(
                                    khanary::micronaut::capture_difference(a.pattern, b.pattern,
                                                                           width)) == expected);
            }
        }
    });
}
