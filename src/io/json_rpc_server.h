#pragma once
#include <string>
#include <unordered_map>
#include <functional>
#include <thread>
#include <atomic>
#include "../util/json.h"

namespace khanary::io {

using RpcHandler = std::function<util::JsonPtr(const util::JsonPtr& params)>;

// Minimal HTTP/1.1 JSON-RPC 2.0 server on port 8787
class JsonRpcServer {
public:
    explicit JsonRpcServer(uint16_t port = 8787);
    ~JsonRpcServer();

    // Register a method handler
    void register_method(const std::string& method, RpcHandler handler);

    // Start listening (non-blocking; spawns accept thread)
    bool start();

    // Stop server
    void stop();

    bool is_running() const { return running_; }

private:
    uint16_t port_;
    int      listen_fd_ = -1;
    std::atomic<bool> running_{false};
    std::thread accept_thread_;
    std::unordered_map<std::string, RpcHandler> methods_;

    void accept_loop();
    void handle_client(int fd);
    std::string dispatch_rpc(const std::string& body);
    static std::string http_response(int status, const std::string& body);
};

} // namespace khanary::io
