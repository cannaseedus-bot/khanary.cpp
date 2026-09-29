#include "factory.h"
#include "../io/runtime_attention.h"
#include "../util/json.h"
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <unordered_set>

namespace khanary::micronaut {
using namespace khanary::util;

namespace {

static float clamp01(float value, float fallback = 1.0f) {
    if (!std::isfinite(value)) return fallback;
    return std::clamp(value, 0.0f, 1.0f);
}

static std::vector<std::string> dedupe_non_empty(const std::vector<std::string>& in) {
    std::vector<std::string> out;
    out.reserve(in.size());
    std::unordered_set<std::string> seen;
    for (const auto& item : in) {
        if (item.empty()) continue;
        if (seen.insert(item).second) out.push_back(item);
    }
    return out;
}

static bool should_replace_word_candidate(
    bool incoming_phrase,
    size_t incoming_len,
    const std::string& incoming_key,
    bool current_phrase,
    size_t current_len,
    const std::string& current_key)
{
    if (current_key.empty()) return true;
    if (incoming_phrase != current_phrase) return incoming_phrase;
    if (incoming_len != current_len) return incoming_len < current_len;
    return incoming_key < current_key;
}

constexpr std::array<char, 8> kFactoryBinMagic = {'K','F','A','C','B','I','N','1'};
constexpr uint32_t kFactoryBinVersion = 1u;

template <typename T>
static bool write_pod(std::ofstream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    return static_cast<bool>(out);
}

template <typename T>
static bool read_pod(std::ifstream& in, T* value) {
    in.read(reinterpret_cast<char*>(value), sizeof(T));
    return static_cast<bool>(in);
}

static bool write_string_blob(std::ofstream& out, const std::string& value) {
    const uint32_t sz = static_cast<uint32_t>(value.size());
    if (!write_pod(out, sz)) return false;
    if (sz == 0) return true;
    out.write(value.data(), sz);
    return static_cast<bool>(out);
}

static bool read_string_blob(std::ifstream& in, std::string* value) {
    uint32_t sz = 0;
    if (!read_pod(in, &sz)) return false;
    value->assign(sz, '\0');
    if (sz == 0) return true;
    in.read(value->data(), sz);
    return static_cast<bool>(in);
}

} // namespace

Factory::Factory(MicronautRegistry& reg, float confidence_floor, float tfidf_accept)
    : reg_(reg),
      confidence_floor_(clamp01(confidence_floor, 0.7f)),
      tfidf_accept_(clamp01(tfidf_accept, 0.42f))
{}

float Factory::lookup_wcr_gate(float key_confidence, float lexical_similarity) {
    const double gate = io::compute_wcr_score({
        1.0,
        std::clamp(static_cast<double>(key_confidence), 0.0, 1.0),
        std::clamp(static_cast<double>(lexical_similarity), 0.0, 1.0)
    });
    return static_cast<float>(gate);
}

kuhul::Phase Factory::op_phase(FactoryOp op) const {
    switch (op) {
        case FactoryOp::Create:  return kuhul::Phase::Wo;
        case FactoryOp::Evolve:  return kuhul::Phase::Yax;
        case FactoryOp::List:    return kuhul::Phase::Pop;
        case FactoryOp::Read:    return kuhul::Phase::Pop;
        case FactoryOp::Edit:    return kuhul::Phase::Sek;
        case FactoryOp::Patch:   return kuhul::Phase::Sek;
        case FactoryOp::Status:  return kuhul::Phase::Chen;
        case FactoryOp::Delete:  return kuhul::Phase::Xul;
    }
    return kuhul::Phase::Pop;
}

std::optional<FactoryCandidate> Factory::propose_create(
    const std::string& id,
    const std::string& description,
    kuhul::Phase primary_phase,
    const std::vector<std::string>& capabilities,
    const std::vector<std::string>& governs)
{
    if (id.empty() || reg_.has(id)) return std::nullopt;
    MicronautEntry entry;
    entry.id            = id;
    entry.description   = description;
    entry.primary_phase = primary_phase;
    entry.capabilities  = dedupe_non_empty(capabilities);
    for (const auto& g : governs)
        if (!g.empty()) entry.governs.insert(g);
    entry.confidence = 1.0f;
    entry.active     = true;
    return FactoryCandidate{
        FactoryCandidate::Op::Create,
        id,
        std::move(entry),
        "micronaut-registry",
        "create:" + id
    };
}

std::optional<FactoryCandidate> Factory::propose_evolve(
    const std::string& id,
    const std::vector<std::string>& new_capabilities,
    const std::vector<std::string>& new_governs)
{
    if (id.empty()) return std::nullopt;
    const auto* e = reg_.get(id);
    if (!e) return std::nullopt;
    // Snapshot merged state — no reg mutation here.
    MicronautEntry snapshot = *e;
    auto merged = dedupe_non_empty(snapshot.capabilities);
    for (const auto& c : dedupe_non_empty(new_capabilities))
        if (std::find(merged.begin(), merged.end(), c) == merged.end())
            merged.push_back(c);
    snapshot.capabilities = std::move(merged);
    for (const auto& g : new_governs)
        if (!g.empty()) snapshot.governs.insert(g);
    return FactoryCandidate{
        FactoryCandidate::Op::Evolve,
        id,
        std::move(snapshot),
        "micronaut-registry",
        "evolve:" + id
    };
}

FactoryResult Factory::commit(FactoryCandidate candidate, kuhul::CommitAuthority authority)
{
    if (authority.scope() != candidate.scope)
        return {false, candidate.id, "commit rejected: scope mismatch"};
    if (authority.proposal_identity() != candidate.proposal_identity)
        return {false, candidate.id, "commit rejected: identity mismatch"};
    const std::string id = candidate.id;
    if (candidate.op == FactoryCandidate::Op::Create) {
        if (reg_.has(id)) return {false, id, "already exists"};
        reg_.add(std::move(candidate.entry));
        return {true, id, "created"};
    }
    if (candidate.op == FactoryCandidate::Op::Edit) {
        auto* e = reg_.get_mut(id);
        if (!e) return {false, id, "not found (stale candidate)"};
        e->capabilities = std::move(candidate.entry.capabilities);
        return {true, id, "edited"};
    }
    if (candidate.op == FactoryCandidate::Op::Patch) {
        auto* e = reg_.get_mut(id);
        if (!e) return {false, id, "not found (stale candidate)"};
        e->confidence = candidate.entry.confidence;
        return {true, id, "patched"};
    }
    if (candidate.op == FactoryCandidate::Op::Remove) {
        const auto* e = reg_.get(id);
        if (!e) return {false, id, "not found (stale candidate)"};
        if (e->confidence < confidence_floor_)
            return {false, id, "confidence below floor (" + std::to_string(e->confidence) + ")"};
        if (!reg_.remove(id, confidence_floor_))
            return {false, id, "delete rejected by registry guard"};
        return {true, id, "deleted"};
    }
    // Evolve: replace entry and re-register gravity nodes.
    auto* e = reg_.get_mut(id);
    if (!e) return {false, id, "not found (stale candidate)"};
    *e = std::move(candidate.entry);
    reg_.gravity().register_nodes(
        id,
        e->governs,
        e->primary_phase,
        e->vector_meta.x,
        e->vector_meta.y,
        e->vector_meta.z,
        e->vector_meta.declared);
    return {true, id, "evolved"};
}

std::vector<std::string> Factory::list() const {
    return reg_.ids();
}

const MicronautEntry* Factory::read(const std::string& id) const {
    return reg_.get(id);
}

std::optional<FactoryCandidate> Factory::propose_edit(
    const std::string& id,
    const std::vector<std::string>& capabilities)
{
    if (id.empty()) return std::nullopt;
    const auto* e = reg_.get(id);
    if (!e) return std::nullopt;
    MicronautEntry snapshot = *e;
    snapshot.capabilities = dedupe_non_empty(capabilities);
    return FactoryCandidate{
        FactoryCandidate::Op::Edit,
        id,
        std::move(snapshot),
        "micronaut-registry",
        "edit:" + id
    };
}

std::optional<FactoryCandidate> Factory::propose_patch(
    const std::string& id, float confidence)
{
    if (id.empty()) return std::nullopt;
    const auto* e = reg_.get(id);
    if (!e) return std::nullopt;
    MicronautEntry snapshot = *e;
    snapshot.confidence = clamp01(confidence, e->confidence);
    return FactoryCandidate{
        FactoryCandidate::Op::Patch,
        id,
        std::move(snapshot),
        "micronaut-registry",
        "patch:" + id
    };
}

FactoryResult Factory::status(const std::string& id) const {
    const auto* e = reg_.get(id);
    if (!e) return {false, id, "not found"};
    return {true, id, e->active ? "active" : "inactive"};
}

std::optional<FactoryCandidate> Factory::propose_remove(const std::string& id)
{
    if (id.empty()) return std::nullopt;
    const auto* e = reg_.get(id);
    if (!e) return std::nullopt;
    if (e->confidence < confidence_floor_) return std::nullopt;
    return FactoryCandidate{
        FactoryCandidate::Op::Remove,
        id,
        *e,
        "micronaut-registry",
        "remove:" + id
    };
}

// ── Agent key store ──────────────────────────────────────────────────────────

// Extract significant words (≥5 chars, alphanumeric+hyphen) from a lowercased string
static std::string lower_ascii(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

static std::string normalize_hash_token(std::string s) {
    s = lower_ascii(std::move(s));
    if (s.rfind("sha256:", 0) == 0) s.erase(0, 7);
    if (s.rfind("hash:", 0) == 0) s.erase(0, 5);
    if (s.rfind("0x", 0) == 0) s.erase(0, 2);
    if (s.empty()) return {};
    for (unsigned char c : s) {
        if (!std::isxdigit(c)) return {};
    }
    return s;
}

static std::string fnv1a64_hex(const std::string& s) {
    uint64_t h = 14695981039346656037ull;
    for (unsigned char c : s) {
        h ^= static_cast<uint64_t>(c);
        h *= 1099511628211ull;
    }
    std::ostringstream ss;
    ss << std::hex << std::setfill('0') << std::setw(16) << h;
    return ss.str();
}

static std::string derive_hash_token(const std::string& lower_key, const std::string& hash_hint) {
    auto hint = normalize_hash_token(hash_hint);
    if (!hint.empty()) return hint;
    return fnv1a64_hex(lower_key);
}

static std::vector<std::string> key_words(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (unsigned char c : s) {
        if (std::isalnum(c) || c == '-') {
            cur += static_cast<char>(std::tolower(c));
        } else if (!cur.empty()) {
            if (cur.size() >= 5) out.push_back(cur);
            cur.clear();
        }
    }
    if (cur.size() >= 5) out.push_back(cur);
    return out;
}

static std::vector<std::string> vector_terms(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (unsigned char c : s) {
        if (std::isalnum(c) || c == '-') {
            cur += static_cast<char>(std::tolower(c));
        } else if (!cur.empty()) {
            if (cur.size() >= 2) out.push_back(cur);
            cur.clear();
        }
    }
    if (cur.size() >= 2) out.push_back(cur);
    return out;
}

void Factory::record_key(const std::string& key, const std::string& response, const std::string& hash_hint) {
    if (key.empty() || response.empty()) return;

    std::lock_guard<std::mutex> lk(key_mtx_);
    std::string kl = lower_ascii(key);
    for (auto it = key_store_.begin(); it != key_store_.end();) {
        if (lower_ascii(it->first) == kl) it = key_store_.erase(it);
        else ++it;
    }
    key_store_[key] = response;
    key_lower_[kl] = response;

    const std::string hash_token = derive_hash_token(kl, hash_hint);
    if (auto old = key_hash_of_key_.find(kl); old != key_hash_of_key_.end() && old->second != hash_token) {
        auto bit = key_hash_idx_.find(old->second);
        if (bit != key_hash_idx_.end()) {
            auto& vec = bit->second;
            vec.erase(std::remove(vec.begin(), vec.end(), kl), vec.end());
            if (vec.empty()) key_hash_idx_.erase(bit);
        }
    }
    key_hash_of_key_[kl] = hash_token;
    auto& hash_bucket = key_hash_idx_[hash_token];
    if (std::find(hash_bucket.begin(), hash_bucket.end(), kl) == hash_bucket.end())
        hash_bucket.push_back(kl);

    auto hit_it = key_hits_.find(kl);
    if (hit_it == key_hits_.end()) {
        key_hits_[kl] = 0;
        key_confidence_[kl] = 0.5f;
    }

    if (bulk_mode_) {
        word_index_dirty_ = true;
    } else {
        rebuild_word_index_locked();
        word_index_dirty_ = false;
    }
}

std::optional<std::string> Factory::lookup_key(const std::string& query) {
    if (query.empty()) return std::nullopt;

    std::lock_guard<std::mutex> lk(key_mtx_);
    auto touch_hit = [this](const std::string& lower_key) -> std::optional<std::string> {
        auto kv = key_lower_.find(lower_key);
        if (kv == key_lower_.end()) return std::nullopt;
        uint64_t& hits = key_hits_[lower_key];
        ++hits;
        if (!key_confidence_.count(lower_key)) key_confidence_[lower_key] = 0.5f;
        return kv->second;
    };

    // 1. Case-sensitive exact match
    auto it = key_store_.find(query);
    if (it != key_store_.end()) {
        if (auto hit = touch_hit(lower_ascii(it->first))) return hit;
    }

    // Build lowercase query once
    std::string qlow = lower_ascii(query);
    if (qlow.empty()) return std::nullopt;

    // 2. Case-insensitive exact match
    if (auto hit = touch_hit(qlow)) return hit;

    // 2b. Explicit hash query: hash:<hex>
    if (qlow.rfind("hash:", 0) == 0) {
        const auto hash_token = normalize_hash_token(qlow.substr(5));
        if (!hash_token.empty()) {
            auto hb = key_hash_idx_.find(hash_token);
            if (hb != key_hash_idx_.end()) {
                for (const auto& candidate : hb->second) {
                    if (auto hit = touch_hit(candidate)) return hit;
                }
            }
        }
    }

    // 2c. Hash-accelerated exact key check (collision-safe by exact lower-key verify)
    {
        const auto qhash = fnv1a64_hex(qlow);
        auto hb = key_hash_idx_.find(qhash);
        if (hb != key_hash_idx_.end()) {
            for (const auto& candidate : hb->second) {
                if (candidate != qlow) continue;
                if (auto hit = touch_hit(candidate)) return hit;
            }
        }
    }

    // 3. Case-insensitive substring scan — phrase keys only (contain space or hyphen),
    //    matched at word boundaries. Single-word keys are handled by pass 4.
    size_t best_len = 0;
    std::string best_phrase_key;
    std::optional<std::string> best_phrase_key_opt;
    for (const auto& [kl, v] : key_lower_) {
        (void)v;
        if (kl.size() < 8) continue;
        if (kl.find(' ') == std::string::npos && kl.find('-') == std::string::npos) continue;
        size_t pos = qlow.find(kl);
        if (pos == std::string::npos) continue;
        bool pre  = (pos == 0) || !std::isalnum(static_cast<unsigned char>(qlow[pos - 1]));
        bool post = (pos + kl.size() >= qlow.size()) ||
                    !std::isalnum(static_cast<unsigned char>(qlow[pos + kl.size()]));
        if (!pre || !post) continue;
        if (kl.size() > best_len || (kl.size() == best_len && (best_phrase_key.empty() || kl < best_phrase_key))) {
            best_len = kl.size();
            best_phrase_key = kl;
            best_phrase_key_opt = kl;
        }
    }
    if (best_phrase_key_opt) {
        if (auto hit = touch_hit(*best_phrase_key_opt)) return hit;
    }

    // 4. Word-index scan — any significant word from the query hits a stored key
    for (const auto& w : key_words(qlow)) {
        auto wk = key_word_key_.find(w);
        if (wk == key_word_key_.end()) continue;
        if (auto hit = touch_hit(wk->second)) return hit;
    }

    // 5. Vector-space fallback (tf-idf cosine) over stored keys.
    const auto query_terms = vector_terms(qlow);
    if (!query_terms.empty() && !key_lower_.empty()) {
        struct KeyDoc {
            const std::string* key = nullptr;
            const std::string* response = nullptr;
            std::unordered_map<std::string, float> tf;
        };

        std::unordered_map<std::string, float> query_tf;
        for (const auto& tok : query_terms) query_tf[tok] += 1.0f;

        std::vector<KeyDoc> docs;
        docs.reserve(key_lower_.size());
        std::unordered_map<std::string, size_t> doc_freq;
        for (const auto& [kl, resp] : key_lower_) {
            auto terms = vector_terms(kl);
            if (terms.empty()) continue;
            KeyDoc doc;
            doc.key = &kl;
            doc.response = &resp;
            for (const auto& tok : terms) doc.tf[tok] += 1.0f;
            if (doc.tf.empty()) continue;
            std::unordered_set<std::string> uniq;
            for (const auto& [tok, count] : doc.tf)
                if (count > 0.0f) uniq.insert(tok);
            for (const auto& tok : uniq) ++doc_freq[tok];
            docs.push_back(std::move(doc));
        }

        if (!docs.empty()) {
            auto idf = [&](const std::string& tok) -> float {
                const auto it = doc_freq.find(tok);
                const double df = (it == doc_freq.end()) ? 0.0 : static_cast<double>(it->second);
                const double n = static_cast<double>(docs.size());
                return static_cast<float>(std::log((1.0 + n) / (1.0 + df)) + 1.0);
            };

            std::unordered_map<std::string, float> query_tfidf;
            float query_norm_sq = 0.0f;
            for (const auto& [tok, tf] : query_tf) {
                const float w = tf * idf(tok);
                query_tfidf[tok] = w;
                query_norm_sq += w * w;
            }
            const float query_norm = std::sqrt(std::max(0.0f, query_norm_sq));

            if (query_norm > 1e-6f) {
                float best_sim = 0.0f;
                std::string best_key;
                const std::string* best_response = nullptr;
                constexpr float kEps = 1e-6f;
                for (const auto& doc : docs) {
                    float dot = 0.0f;
                    float doc_norm_sq = 0.0f;
                    for (const auto& [tok, tfv] : doc.tf) {
                        const float w = tfv * idf(tok);
                        doc_norm_sq += w * w;
                        auto qit = query_tfidf.find(tok);
                        if (qit != query_tfidf.end()) dot += qit->second * w;
                    }
                    const float doc_norm = std::sqrt(std::max(0.0f, doc_norm_sq));
                    if (doc_norm <= 1e-6f) continue;
                    const float sim = dot / (query_norm * doc_norm);
                    const std::string doc_key = doc.key ? *doc.key : std::string{};
                    if (sim > best_sim + kEps ||
                        (std::fabs(sim - best_sim) <= kEps && !best_key.empty() && doc_key < best_key)) {
                        best_sim = sim;
                        best_key = doc_key;
                        best_response = doc.response;
                    }
                }
                const float best_key_conf = (!best_key.empty() && key_confidence_.count(best_key))
                    ? clamp01(key_confidence_.at(best_key), 0.5f)
                    : 0.5f;
                const float wcr_gate = lookup_wcr_gate(best_key_conf, best_sim);
                if (best_response && (best_sim * wcr_gate) >= tfidf_accept_) {
                    if (!best_key.empty()) {
                        if (auto hit = touch_hit(best_key)) return hit;
                    }
                    return *best_response;
                }
            }
        }
    }

    return std::nullopt;
}

void Factory::rebuild_word_index_locked() {
    key_word_idx_.clear();
    key_word_key_.clear();
    key_word_len_.clear();
    key_word_phrase_.clear();

    for (const auto& [kl, response] : key_lower_) {
        const size_t klen = kl.size();
        const bool is_phrase = (kl.find(' ') != std::string::npos);
        for (const auto& word : key_words(kl)) {
            auto lit = key_word_len_.find(word);
            if (lit == key_word_len_.end()) {
                key_word_idx_[word] = response;
                key_word_key_[word] = kl;
                key_word_len_[word] = klen;
                key_word_phrase_[word] = is_phrase;
                continue;
            }
            const bool cur_phrase = key_word_phrase_.count(word) && key_word_phrase_.at(word);
            const std::string current_key =
                key_word_key_.count(word) ? key_word_key_.at(word) : std::string{};
            if (should_replace_word_candidate(
                    is_phrase, klen, kl,
                    cur_phrase, lit->second, current_key)) {
                key_word_idx_[word] = response;
                key_word_key_[word] = kl;
                key_word_len_[word] = klen;
                key_word_phrase_[word] = is_phrase;
            }
        }
    }
}

std::vector<std::string> Factory::list_keys() const {
    std::lock_guard<std::mutex> lk(key_mtx_);
    std::vector<std::string> out;
    out.reserve(key_store_.size());
    for (const auto& [k, _] : key_store_) out.push_back(k);
    std::sort(out.begin(), out.end());
    return out;
}

std::optional<std::string> Factory::peek_key(const std::string& key) const {
    if (key.empty()) return std::nullopt;
    std::lock_guard<std::mutex> lk(key_mtx_);

    if (auto it = key_store_.find(key); it != key_store_.end())
        return it->second;

    const auto kl = lower_ascii(key);
    if (auto it = key_lower_.find(kl); it != key_lower_.end())
        return it->second;

    return std::nullopt;
}

std::string Factory::key_id_for(const std::string& key) const {
    if (key.empty()) return {};
    std::lock_guard<std::mutex> lk(key_mtx_);

    const auto kl = lower_ascii(key);
    if (kl.empty()) return {};

    if (auto it = key_hash_of_key_.find(kl); it != key_hash_of_key_.end() && !it->second.empty())
        return it->second;
    return derive_hash_token(kl, "");
}

std::vector<std::string> Factory::keys_for_id(const std::string& key_id) const {
    if (key_id.empty()) return {};

    std::string token = key_id;
    if (token.rfind("hash:", 0) == 0) token.erase(0, 5);
    token = normalize_hash_token(token);
    if (token.empty()) return {};

    std::lock_guard<std::mutex> lk(key_mtx_);
    auto it = key_hash_idx_.find(token);
    if (it == key_hash_idx_.end()) return {};

    std::unordered_map<std::string, std::string> display_by_lower;
    display_by_lower.reserve(key_store_.size());
    for (const auto& [orig, _] : key_store_) {
        const auto lk_key = lower_ascii(orig);
        auto dit = display_by_lower.find(lk_key);
        if (dit == display_by_lower.end() || orig < dit->second)
            display_by_lower[lk_key] = orig;
    }

    std::vector<std::string> out;
    out.reserve(it->second.size());
    for (const auto& lower_key : it->second) {
        if (auto dit = display_by_lower.find(lower_key); dit != display_by_lower.end())
            out.push_back(dit->second);
        else
            out.push_back(lower_key);
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

uint64_t Factory::key_hits(const std::string& key) const {
    if (key.empty()) return 0;
    std::lock_guard<std::mutex> lk(key_mtx_);
    const auto kl = lower_ascii(key);
    auto it = key_hits_.find(kl);
    return it != key_hits_.end() ? it->second : 0;
}

float Factory::key_confidence(const std::string& key) const {
    if (key.empty()) return 0.5f;
    std::lock_guard<std::mutex> lk(key_mtx_);
    const auto kl = lower_ascii(key);
    auto it = key_confidence_.find(kl);
    if (it != key_confidence_.end()) return clamp01(it->second, 0.5f);
    return 0.5f;
}

void Factory::begin_bulk_update() {
    std::lock_guard<std::mutex> lk(key_mtx_);
    bulk_mode_ = true;
}

void Factory::end_bulk_update() {
    std::lock_guard<std::mutex> lk(key_mtx_);
    bulk_mode_ = false;
    if (word_index_dirty_) {
        rebuild_word_index_locked();
        word_index_dirty_ = false;
    }
}

bool Factory::persist(kuhul::CommitAuthority authority,
                      const std::string& json_path, const std::string& bin_path) {
    (void)std::move(authority); // possession proves authorization; consumed here
    bool ok = save_state(json_path);
    return save_state_binary(bin_path) && ok;
}

bool Factory::persist_binary(kuhul::CommitAuthority authority, const std::string& bin_path) {
    (void)std::move(authority);
    return save_state_binary(bin_path);
}

bool Factory::save_state(const std::string& path) const {
    if (path.empty()) return false;

    std::lock_guard<std::mutex> lk(key_mtx_);
    JsonObject root;
    root["version"] = json_number(1.0);
    root["tfidf_accept"] = json_number(tfidf_accept_);

    JsonArray entries;
    entries.reserve(key_store_.size());
    for (const auto& [key, response] : key_store_) {
        const auto kl = lower_ascii(key);
        JsonObject item;
        item["key"] = json_string(key);
        item["response"] = json_string(response);
        if (auto h = key_hash_of_key_.find(kl); h != key_hash_of_key_.end())
            item["hash"] = json_string(h->second);
        if (auto hc = key_hits_.find(kl); hc != key_hits_.end())
            item["hits"] = json_number(static_cast<double>(hc->second));
        if (auto cf = key_confidence_.find(kl); cf != key_confidence_.end())
            // clamp01's second argument is a FALLBACK for a non-finite value, not a
            // lower bound: stored evidence confidence is written verbatim either
            // way. 0.0f keeps a corrupt/missing value neutral instead of inferring
            // it from usage.
            item["confidence"] = json_number(clamp01(cf->second, 0.0f));
        entries.push_back(json_object(std::move(item)));
    }
    root["entries"] = json_array(std::move(entries));

    std::ofstream out(path);
    if (!out) return false;
    out << stringify(json_object(std::move(root)), true);
    return true;
}

bool Factory::load_state(const std::string& path) {
    if (path.empty()) return false;

    std::ifstream in(path);
    if (!in) return false;
    std::ostringstream ss;
    ss << in.rdbuf();
    auto root = parse(ss.str());
    if (!root || !root->is_object()) return false;

    const auto* entries = root->get("entries");
    if (!entries || !entries->is_array()) return false;

    std::lock_guard<std::mutex> lk(key_mtx_);
    key_store_.clear();
    key_lower_.clear();
    key_hash_of_key_.clear();
    key_hash_idx_.clear();
    key_word_idx_.clear();
    key_word_key_.clear();
    key_word_len_.clear();
    key_word_phrase_.clear();
    key_hits_.clear();
    key_confidence_.clear();

    if (auto tf = root->get_number("tfidf_accept"))
        tfidf_accept_ = clamp01(static_cast<float>(*tf), tfidf_accept_);

    for (const auto& ev : entries->as_array()) {
        if (!ev || !ev->is_object()) continue;
        auto key = ev->get_string("key");
        auto response = ev->get_string("response");
        if (!key || !response || key->empty() || response->empty()) continue;

        const std::string kl = lower_ascii(*key);
        key_store_[*key] = *response;
        key_lower_[kl] = *response;

        std::string hash_token;
        if (auto h = ev->get_string("hash")) hash_token = normalize_hash_token(*h);
        if (hash_token.empty()) hash_token = derive_hash_token(kl, "");
        key_hash_of_key_[kl] = hash_token;
        key_hash_idx_[hash_token].push_back(kl);

        uint64_t hits = 0;
        if (auto hv = ev->get_number("hits")) {
            const double raw = *hv;
            if (std::isfinite(raw) && raw > 0.0) {
                hits = static_cast<uint64_t>(raw);
            }
        }
        key_hits_[kl] = hits;
        // clamp01's second argument is a fallback for non-finite input, not a
        // floor. Usage is stored separately from evidence confidence.
        if (auto cf = ev->get_number("confidence"))
            key_confidence_[kl] = clamp01(static_cast<float>(*cf), 0.0f);
        else
            key_confidence_[kl] = 0.5f;
    }

    for (auto& [hash, bucket] : key_hash_idx_) {
        std::sort(bucket.begin(), bucket.end());
        bucket.erase(std::unique(bucket.begin(), bucket.end()), bucket.end());
    }

    rebuild_word_index_locked();
    word_index_dirty_ = false;
    return true;
}

bool Factory::save_state_binary(const std::string& path) const {
    if (path.empty()) return false;

    std::lock_guard<std::mutex> lk(key_mtx_);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) return false;

    out.write(kFactoryBinMagic.data(), static_cast<std::streamsize>(kFactoryBinMagic.size()));
    if (!out) return false;
    if (!write_pod(out, kFactoryBinVersion)) return false;
    if (!write_pod(out, tfidf_accept_)) return false;

    const uint64_t count = static_cast<uint64_t>(key_store_.size());
    if (!write_pod(out, count)) return false;

    for (const auto& [key, response] : key_store_) {
        const auto kl = lower_ascii(key);
        std::string hash_token;
        if (auto h = key_hash_of_key_.find(kl); h != key_hash_of_key_.end())
            hash_token = h->second;
        if (hash_token.empty()) hash_token = derive_hash_token(kl, "");

        uint64_t hits = 0;
        if (auto hc = key_hits_.find(kl); hc != key_hits_.end())
            hits = hc->second;

        // As in the JSON path: the fallback is a neutral prior (0.5), not a
        // usage-derived value. Usage (hits) is stored in its own field and must
        // not influence the evidence-based confidence column.
        float confidence = 0.5f;
        if (auto cf = key_confidence_.find(kl); cf != key_confidence_.end())
            confidence = clamp01(cf->second, 0.5f);

        if (!write_string_blob(out, key)) return false;
        if (!write_string_blob(out, response)) return false;
        if (!write_string_blob(out, hash_token)) return false;
        if (!write_pod(out, hits)) return false;
        if (!write_pod(out, confidence)) return false;
    }
    return static_cast<bool>(out);
}

bool Factory::load_state_binary(const std::string& path) {
    if (path.empty()) return false;

    std::ifstream in(path, std::ios::binary);
    if (!in) return false;

    std::array<char, kFactoryBinMagic.size()> magic{};
    in.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!in || magic != kFactoryBinMagic) return false;

    uint32_t version = 0;
    if (!read_pod(in, &version)) return false;
    if (version != kFactoryBinVersion) return false;

    float tfidf_accept = tfidf_accept_;
    if (!read_pod(in, &tfidf_accept)) return false;

    uint64_t count = 0;
    if (!read_pod(in, &count)) return false;

    std::lock_guard<std::mutex> lk(key_mtx_);
    key_store_.clear();
    key_lower_.clear();
    key_hash_of_key_.clear();
    key_hash_idx_.clear();
    key_word_idx_.clear();
    key_word_key_.clear();
    key_word_len_.clear();
    key_word_phrase_.clear();
    key_hits_.clear();
    key_confidence_.clear();
    tfidf_accept_ = clamp01(tfidf_accept, tfidf_accept_);

    for (uint64_t i = 0; i < count; ++i) {
        std::string key;
        std::string response;
        std::string hash_token;
        uint64_t hits = 0;
        float confidence = 0.5f;

        if (!read_string_blob(in, &key)) return false;
        if (!read_string_blob(in, &response)) return false;
        if (!read_string_blob(in, &hash_token)) return false;
        if (!read_pod(in, &hits)) return false;
        if (!read_pod(in, &confidence)) return false;

        if (key.empty() || response.empty()) continue;

        const std::string kl = lower_ascii(key);
        key_store_[key] = response;
        key_lower_[kl] = response;

        hash_token = normalize_hash_token(hash_token);
        if (hash_token.empty()) hash_token = derive_hash_token(kl, "");
        key_hash_of_key_[kl] = hash_token;
        key_hash_idx_[hash_token].push_back(kl);

        key_hits_[kl] = hits;
        // As in the JSON loader: the second argument is a fallback, not a floor.
        // Usage is stored separately from evidence confidence.
        key_confidence_[kl] = clamp01(confidence, 0.0f);
    }

    for (auto& [hash, bucket] : key_hash_idx_) {
        std::sort(bucket.begin(), bucket.end());
        bucket.erase(std::unique(bucket.begin(), bucket.end()), bucket.end());
    }

    rebuild_word_index_locked();
    word_index_dirty_ = false;
    return true;
}

bool Factory::remove_key(const std::string& key) {
    std::lock_guard<std::mutex> lk(key_mtx_);
    bool removed = key_store_.erase(key) > 0;
    const std::string kl = lower_ascii(key);

    if (!kl.empty()) {
        for (auto it = key_store_.begin(); it != key_store_.end();) {
            if (lower_ascii(it->first) == kl) {
                it = key_store_.erase(it);
                removed = true;
            } else {
                ++it;
            }
        }
    }

    key_lower_.erase(kl);
    key_hits_.erase(kl);
    key_confidence_.erase(kl);

    auto hit = key_hash_of_key_.find(kl);
    if (hit != key_hash_of_key_.end()) {
        auto bit = key_hash_idx_.find(hit->second);
        if (bit != key_hash_idx_.end()) {
            auto& vec = bit->second;
            vec.erase(std::remove(vec.begin(), vec.end(), kl), vec.end());
            if (vec.empty()) key_hash_idx_.erase(bit);
        }
        key_hash_of_key_.erase(hit);
    }
    if (bulk_mode_) {
        word_index_dirty_ = true;
    } else {
        rebuild_word_index_locked();
        word_index_dirty_ = false;
    }
    return removed;
}

} // namespace khanary::micronaut
