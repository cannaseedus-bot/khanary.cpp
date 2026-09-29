#pragma once
#include "micronaut_registry.h"
#include "../kuhul/phase_engine.h"
#include "../kuhul/xul_gate.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include <mutex>
#include <cstdint>
#include <utility>

namespace khanary::micronaut {

enum class FactoryOp {
    Create,   // Wo phase
    Evolve,   // Yax phase
    List,     // Pop phase
    Read,     // Pop phase
    Edit,     // Sek phase
    Patch,    // Sek phase
    Status,   // Ch'en phase
    Delete,   // Xul phase (requires confidence >= floor)
};

struct FactoryResult {
    bool        ok;
    std::string id;
    std::string message;
};

// Proposal record returned by propose_create/propose_evolve/propose_edit/
// propose_patch/propose_remove.
// Carries the full mutation snapshot and the scope/identity pair that commit()
// uses to validate the CommitAuthority matches this exact proposal.
struct FactoryCandidate {
    enum class Op { Create, Evolve, Edit, Patch, Remove };
    Op             op;
    std::string    id;
    MicronautEntry entry;          // Create/Evolve: full snapshot; Edit/Patch: updated snapshot; Remove: current snapshot
    std::string    scope;
    std::string    proposal_identity;
};

class Factory {
public:
    Factory(MicronautRegistry& reg,
            float confidence_floor = 0.7f,
            float tfidf_accept = 0.42f);

    // Propose a new micronaut without touching the registry.
    // Returns nullopt if the id is empty or already exists.
    std::optional<FactoryCandidate> propose_create(
        const std::string& id,
        const std::string& description,
        kuhul::Phase primary_phase,
        const std::vector<std::string>& capabilities,
        const std::vector<std::string>& governs);

    // Propose adding capabilities/governs to an existing micronaut.
    // Returns nullopt if the id is missing or not found.
    std::optional<FactoryCandidate> propose_evolve(
        const std::string& id,
        const std::vector<std::string>& new_capabilities,
        const std::vector<std::string>& new_governs);

    // Apply a proposal to the registry. The CommitAuthority is consumed
    // and must match the candidate's scope and proposal_identity exactly.
    FactoryResult commit(FactoryCandidate candidate, kuhul::CommitAuthority authority);

    // List all micronaut ids
    std::vector<std::string> list() const;

    // Read a micronaut definition
    const MicronautEntry* read(const std::string& id) const;

    // Propose replacing a micronaut's capability list.
    // Returns nullopt if id is empty or not found.
    std::optional<FactoryCandidate> propose_edit(
        const std::string& id,
        const std::vector<std::string>& capabilities);

    // Propose an evidence-based confidence update.
    // Returns nullopt if id is empty or not found.
    std::optional<FactoryCandidate> propose_patch(
        const std::string& id, float confidence);

    // Status: returns active flag
    FactoryResult status(const std::string& id) const;

    // Propose removing a micronaut (only if confidence >= floor).
    // Returns nullopt if id is empty, not found, or below the confidence floor.
    std::optional<FactoryCandidate> propose_remove(const std::string& id);

    kuhul::Phase op_phase(FactoryOp op) const;

    // ── Agent key store ───────────────────────────────────────────────────
    // Record a key→response pair extracted from a model output.
    // Key is a canonical trigger phrase; response is the stored reply.
    // hash_hint (optional) allows external datasets to supply a stable hash id.
    void record_key(const std::string& key,
                    const std::string& response,
                    const std::string& hash_hint = "");

    // Look up a stored response by exact key match, then prefix match.
    // Returns the response string if found, nullopt otherwise.
    std::optional<std::string> lookup_key(const std::string& query);

    // Returns all registered keys.
    std::vector<std::string> list_keys() const;

    // Read-only exact/prefix-free key lookup without reinforcement updates.
    std::optional<std::string> peek_key(const std::string& key) const;

    // Deterministic key identifier used by factory_keys.bin. Returns lowercase hex.
    std::string key_id_for(const std::string& key) const;

    // Reverse mapping from key identifier (hex/hash token) to known keys.
    std::vector<std::string> keys_for_id(const std::string& key_id) const;

    // Remove a key from the store.
    bool remove_key(const std::string& key);

    // Lookup metadata (reinforcement signal)
    uint64_t key_hits(const std::string& key) const;
    float key_confidence(const std::string& key) const;

    // Bulk mode: defer index rebuild during large imports.
    void begin_bulk_update();
    void end_bulk_update();

    // Restore key store and retrieval metadata.
    bool load_state(const std::string& path);
    bool load_state_binary(const std::string& path);

    // Persist with CommitAuthority — the authority is consumed (move-only).
    // Both json_path and bin_path are written atomically under one authority.
    bool persist(kuhul::CommitAuthority authority,
                 const std::string& json_path, const std::string& bin_path);
    // Persist binary format only (e.g. cache rebuild without a full JSON save).
    bool persist_binary(kuhul::CommitAuthority authority, const std::string& bin_path);

private:
    bool save_state(const std::string& path) const;
    bool save_state_binary(const std::string& path) const;
    MicronautRegistry& reg_;
    float confidence_floor_;

    // agent key → canned response (model-derived)
    mutable std::mutex key_mtx_;
    std::unordered_map<std::string, std::string> key_store_;
    std::unordered_map<std::string, std::string> key_lower_;  // lowercase(key) → value
    std::unordered_map<std::string, std::string> key_hash_of_key_; // lowercase(key) -> hash token
    std::unordered_map<std::string, std::vector<std::string>> key_hash_idx_; // hash token -> lowercase keys
    std::unordered_map<std::string, std::string> key_word_idx_;    // word → value
    std::unordered_map<std::string, std::string> key_word_key_;    // word → lowercase key winner
    std::unordered_map<std::string, size_t>      key_word_len_;    // word → winning key length
    std::unordered_map<std::string, bool>        key_word_phrase_; // word → winning key is multi-word phrase
    std::unordered_map<std::string, uint64_t>    key_hits_;        // lowercase key → lookup hits
    std::unordered_map<std::string, float>       key_confidence_;  // lowercase key → [0,1]
    float tfidf_accept_;
    bool bulk_mode_ = false;
    bool word_index_dirty_ = false;

    void rebuild_word_index_locked();
    static float lookup_wcr_gate(float key_confidence, float lexical_similarity);
};

} // namespace khanary::micronaut
