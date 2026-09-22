#pragma once
#include <thread>
#include <functional>
#include <future>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <stdexcept>

namespace khanary::util {

class ThreadPool {
public:
    explicit ThreadPool(size_t n_threads = 0);
    ~ThreadPool();

    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>>;

    size_t size() const { return workers_.size(); }
    void wait_all();

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mtx_;
    std::condition_variable cv_;
    std::condition_variable done_cv_;
    bool stop_ = false;
    size_t active_ = 0;
};

template<typename F, typename... Args>
auto ThreadPool::submit(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>> {
    using RetT = std::invoke_result_t<F, Args...>;
    auto task = std::make_shared<std::packaged_task<RetT()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );
    std::future<RetT> res = task->get_future();
    {
        std::lock_guard lk(mtx_);
        if (stop_) throw std::runtime_error("ThreadPool: submit after shutdown");
        tasks_.emplace([task]{ (*task)(); });
    }
    cv_.notify_one();
    return res;
}

} // namespace khanary::util
