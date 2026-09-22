#pragma once
#include "tensor_buffer.h"
#include "../config.h"
#include <string>
#include <unordered_map>
#include <list>
#include <memory>
#include <mutex>

namespace khanary::gpu {

class HotSwapManager {
public:
    explicit HotSwapManager(size_t ceiling_bytes = HD4600_RESIDENT_CEILING_BYTES);

    // Make a tensor buffer resident (may evict LRU if needed)
    bool make_resident(const std::string& id, std::shared_ptr<TensorBuffer> buf);

    // Evict at least n_bytes of resident tensors (LRU order)
    size_t evict(size_t n_bytes);

    // Mark as recently used (moves to front of LRU)
    void touch(const std::string& id);

    bool is_resident(const std::string& id) const;
    size_t resident_bytes() const;
    size_t ceiling() const { return ceiling_; }

    // Available headroom
    size_t available() const;

private:
    size_t ceiling_;
    size_t resident_bytes_ = 0;

    struct Entry {
        std::string id;
        std::shared_ptr<TensorBuffer> buf;
    };
    std::list<Entry> lru_;
    std::unordered_map<std::string, std::list<Entry>::iterator> index_;
    mutable std::mutex mtx_;
};

} // namespace khanary::gpu
