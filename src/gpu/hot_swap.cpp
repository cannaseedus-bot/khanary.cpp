#include "hot_swap.h"

namespace khanary::gpu {

HotSwapManager::HotSwapManager(size_t ceiling_bytes) : ceiling_(ceiling_bytes) {}

bool HotSwapManager::make_resident(const std::string& id, std::shared_ptr<TensorBuffer> buf) {
    if (!buf || !buf->is_valid()) return false;
    size_t sz = buf->byte_size();
    std::lock_guard lk(mtx_);
    if (index_.count(id)) {
        // Already resident — just touch
        auto it = index_[id];
        lru_.splice(lru_.begin(), lru_, it);
        return true;
    }
    // Evict if needed
    while (resident_bytes_ + sz > ceiling_ && !lru_.empty()) {
        auto& back = lru_.back();
        resident_bytes_ -= back.buf->byte_size();
        index_.erase(back.id);
        lru_.pop_back();
    }
    if (resident_bytes_ + sz > ceiling_) return false;
    lru_.push_front({id, buf});
    index_[id] = lru_.begin();
    resident_bytes_ += sz;
    return true;
}

size_t HotSwapManager::evict(size_t n_bytes) {
    std::lock_guard lk(mtx_);
    size_t evicted = 0;
    while (evicted < n_bytes && !lru_.empty()) {
        auto& back = lru_.back();
        size_t sz = back.buf->byte_size();
        evicted += sz;
        resident_bytes_ -= sz;
        index_.erase(back.id);
        lru_.pop_back();
    }
    return evicted;
}

void HotSwapManager::touch(const std::string& id) {
    std::lock_guard lk(mtx_);
    auto it = index_.find(id);
    if (it == index_.end()) return;
    lru_.splice(lru_.begin(), lru_, it->second);
}

bool HotSwapManager::is_resident(const std::string& id) const {
    std::lock_guard lk(mtx_);
    return index_.count(id) > 0;
}

size_t HotSwapManager::resident_bytes() const {
    std::lock_guard lk(mtx_);
    return resident_bytes_;
}

size_t HotSwapManager::available() const {
    std::lock_guard lk(mtx_);
    return ceiling_ > resident_bytes_ ? ceiling_ - resident_bytes_ : 0;
}

} // namespace khanary::gpu
