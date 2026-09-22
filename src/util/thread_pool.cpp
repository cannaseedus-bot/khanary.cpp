#include "thread_pool.h"

namespace khanary::util {

ThreadPool::ThreadPool(size_t n_threads) {
    if (n_threads == 0)
        n_threads = std::max(1u, std::thread::hardware_concurrency());
    workers_.reserve(n_threads);
    for (size_t i = 0; i < n_threads; ++i) {
        workers_.emplace_back([this] {
            for (;;) {
                std::function<void()> task;
                {
                    std::unique_lock lk(mtx_);
                    cv_.wait(lk, [this]{ return stop_ || !tasks_.empty(); });
                    if (stop_ && tasks_.empty()) return;
                    task = std::move(tasks_.front());
                    tasks_.pop();
                    ++active_;
                }
                task();
                {
                    std::lock_guard lk(mtx_);
                    --active_;
                }
                done_cv_.notify_all();
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard lk(mtx_);
        stop_ = true;
    }
    cv_.notify_all();
    for (auto& w : workers_) w.join();
}

void ThreadPool::wait_all() {
    std::unique_lock lk(mtx_);
    done_cv_.wait(lk, [this]{ return tasks_.empty() && active_ == 0; });
}

} // namespace khanary::util
