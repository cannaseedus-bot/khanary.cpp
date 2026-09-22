#include "eliza_operators.h"
#include <algorithm>

namespace khanary::micronaut {

void ElizaOperatorBank::init() {
    ops_ = {
        { "CAUSE",       {"because","therefore","since","causes","result","due"},      "CAUSE[{context}] → effect" },
        { "CONTRAST",    {"but","however","although","yet","despite","whereas"},        "CONTRAST[{context}] ↔ opposition" },
        { "DEPENDENCY",  {"requires","depends","needs","prerequisite","if","then"},     "DEPEND[{context}] ← precondition" },
        { "CONSTRAINT",  {"must","cannot","forbidden","limit","restrict","bound"},      "CONSTRAIN[{context}] ∈ bounds" },
        { "EVIDENCE",    {"shows","proves","indicates","demonstrates","evidence","data"},"EVIDENCE[{context}] ⊢ claim" },
        { "SEQUENCE",    {"first","then","next","finally","after","before","step"},     "SEQ[{context}] → ordered" },
        { "BOUNDARY",    {"start","end","begin","stop","boundary","edge","limit"},      "BOUND[{context}] ∂domain" },
        { "STATE",       {"is","are","was","were","state","status","condition"},        "STATE[{context}] = value" },
        { "CONFLICT",    {"conflict","contradiction","inconsistent","oppose","versus"}, "CONFLICT[{context}] ⊗ resolution" },
        { "ALTERNATIVE", {"or","either","alternatively","instead","option","choice"},   "ALT[{context}] ∨ branch" },
        { "INVARIANT",   {"always","never","constant","invariant","unchanged","fixed"}, "INVAR[{context}] ∀ constant" },
        { "TEST",        {"test","check","verify","assert","validate","confirm"},       "TEST[{context}] ? assertion" },
        { "RELATION",    {"related","similar","like","connects","between","maps"},      "RELATE[{context}] ~ neighbor" },
        { "SUBSTITUTE",  {"replace","substitute","swap","instead","equivalent","equal"},"SUBST[{context}] ↦ replacement" },
        { "VERIFY",      {"true","false","correct","wrong","valid","invalid","confirm"},"VERIFY[{context}] ✓ truthvalue" },
        { "DIRECT",      {"go","run","execute","perform","do","apply","invoke"},        "DIRECT[{context}] → action" },
    };
    for (size_t i = 0; i < ops_.size(); ++i)
        name_index_[ops_[i].name] = i;
}

ElizaOperatorBank::ElizaOperatorBank() { init(); }

std::optional<OperatorMatch> ElizaOperatorBank::detect(
    const std::vector<std::string>& tokens) const
{
    OperatorMatch best;
    best.score = -1.0f;

    for (const auto& op : ops_) {
        float score = 0.0f;
        std::vector<std::string> matched;
        for (const auto& cue : op.pop_governs) {
            for (const auto& tok : tokens) {
                std::string lc = tok;
                for (char& c : lc) c = (char)tolower((unsigned char)c);
                if (lc == cue || lc.find(cue) != std::string::npos) {
                    score += 1.0f;
                    matched.push_back(tok);
                    break;
                }
            }
        }
        // Normalize by number of governs
        if (!op.pop_governs.empty())
            score /= (float)op.pop_governs.size();

        if (score > best.score && score > 0.0f) {
            best.score = score;
            best.op = &op;
            best.matched_cues = matched;
        }
    }
    if (!best.op) return std::nullopt;
    return best;
}

std::string ElizaOperatorBank::emit(const ElizaOperator& op, const std::string& context) const {
    std::string tpl = op.sek_directive;
    size_t pos = tpl.find("{context}");
    if (pos != std::string::npos)
        tpl.replace(pos, 9, context.empty() ? "?" : context);
    return tpl;
}

const ElizaOperator* ElizaOperatorBank::find(const std::string& name) const {
    auto it = name_index_.find(name);
    if (it == name_index_.end()) return nullptr;
    return &ops_[it->second];
}

} // namespace khanary::micronaut
