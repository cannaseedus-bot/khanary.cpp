#include "factory.h"

namespace khanary::micronaut {

Factory::Factory(MicronautRegistry& reg, float confidence_floor)
    : reg_(reg), confidence_floor_(confidence_floor)
{}

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

FactoryResult Factory::create(
    const std::string& id,
    const std::string& description,
    kuhul::Phase primary_phase,
    const std::vector<std::string>& capabilities,
    const std::vector<std::string>& governs)
{
    if (reg_.has(id))
        return {false, id, "micronaut '" + id + "' already exists"};

    MicronautEntry entry;
    entry.id           = id;
    entry.description  = description;
    entry.primary_phase= primary_phase;
    entry.capabilities = capabilities;
    entry.governs      = std::unordered_set<std::string>(governs.begin(), governs.end());
    entry.confidence   = 1.0f;
    entry.active       = true;
    reg_.add(std::move(entry));
    return {true, id, "created"};
}

FactoryResult Factory::evolve(
    const std::string& id,
    const std::vector<std::string>& new_capabilities,
    const std::vector<std::string>& new_governs)
{
    auto* e = reg_.get_mut(id);
    if (!e) return {false, id, "not found"};
    for (const auto& c : new_capabilities) e->capabilities.push_back(c);
    for (const auto& g : new_governs)      e->governs.insert(g);
    // Re-register gravity nodes
    reg_.gravity().register_nodes(id, e->governs);
    return {true, id, "evolved"};
}

std::vector<std::string> Factory::list() const {
    return reg_.ids();
}

const MicronautEntry* Factory::read(const std::string& id) const {
    return reg_.get(id);
}

FactoryResult Factory::edit(
    const std::string& id,
    const std::vector<std::string>& capabilities)
{
    auto* e = reg_.get_mut(id);
    if (!e) return {false, id, "not found"};
    e->capabilities = capabilities;
    return {true, id, "edited"};
}

FactoryResult Factory::patch(const std::string& id, float confidence) {
    auto* e = reg_.get_mut(id);
    if (!e) return {false, id, "not found"};
    e->confidence = std::max(0.0f, std::min(1.0f, confidence));
    return {true, id, "patched"};
}

FactoryResult Factory::status(const std::string& id) const {
    const auto* e = reg_.get(id);
    if (!e) return {false, id, "not found"};
    return {true, id, e->active ? "active" : "inactive"};
}

FactoryResult Factory::remove(const std::string& id) {
    const auto* e = reg_.get(id);
    if (!e) return {false, id, "not found"};
    if (e->confidence < confidence_floor_)
        return {false, id, "confidence below floor (" + std::to_string(e->confidence) + ")"};
    reg_.remove(id, confidence_floor_);
    return {true, id, "deleted"};
}

} // namespace khanary::micronaut
