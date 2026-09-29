#include "json_rpc_server.h"
#include <winsock2.h>
#include <ws2tcpip.h>
#include <stdexcept>
#include <sstream>

#pragma comment(lib, "Ws2_32.lib")

namespace khanary::io {

using namespace khanary::util;

JsonRpcServer::JsonRpcServer(uint16_t port) : port_(port) {
    WSADATA wsd{};
    WSAStartup(MAKEWORD(2, 2), &wsd);
}

JsonRpcServer::~JsonRpcServer() { stop(); WSACleanup(); }

void JsonRpcServer::register_method(const std::string& method, RpcHandler h) {
    methods_[method] = std::move(h);
}

bool JsonRpcServer::start() {
    listen_fd_ = (int)socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (listen_fd_ < 0) return false;

    int opt = 1;
    setsockopt((SOCKET)listen_fd_, SOL_SOCKET, SO_REUSEADDR, (char*)&opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(port_);

    if (bind((SOCKET)listen_fd_, (sockaddr*)&addr, sizeof(addr)) < 0) return false;
    if (listen((SOCKET)listen_fd_, 8) < 0) return false;

    running_ = true;
    accept_thread_ = std::thread([this]{ accept_loop(); });
    return true;
}

void JsonRpcServer::stop() {
    running_ = false;
    if (listen_fd_ >= 0) { closesocket((SOCKET)listen_fd_); listen_fd_ = -1; }
    if (accept_thread_.joinable()) accept_thread_.join();
}

void JsonRpcServer::accept_loop() {
    while (running_) {
        sockaddr_in client_addr{};
        int client_len = sizeof(client_addr);
        int client_fd = (int)accept((SOCKET)listen_fd_, (sockaddr*)&client_addr, &client_len);
        if (client_fd < 0) break;
        std::thread([this, client_fd]{ handle_client(client_fd); }).detach();
    }
}

void JsonRpcServer::handle_client(int fd) {
    char buf[65536];
    int n = recv((SOCKET)fd, buf, sizeof(buf) - 1, 0);
    if (n <= 0) { closesocket((SOCKET)fd); return; }
    buf[n] = '\0';
    std::string req(buf, n);

    // Find HTTP body (after \r\n\r\n)
    size_t sep = req.find("\r\n\r\n");
    std::string body = (sep != std::string::npos) ? req.substr(sep + 4) : req;

    std::string resp_body = dispatch_rpc(body);
    std::string resp = http_response(200, resp_body);
    send((SOCKET)fd, resp.data(), (int)resp.size(), 0);
    closesocket((SOCKET)fd);
}

std::string JsonRpcServer::dispatch_rpc(const std::string& body) {
    try {
        auto req = parse(body);
        if (!req || !req->is_object())
            return R"({"jsonrpc":"2.0","error":{"code":-32700,"message":"Parse error"},"id":null})";

        auto method_v = req->get_string("method");
        auto id_v     = req->get("id");
        std::string id_str = id_v ? stringify(*id_v) : "null";

        if (!method_v)
            return R"({"jsonrpc":"2.0","error":{"code":-32600,"message":"Invalid Request"},"id":)" + id_str + "}";

        const auto& method = *method_v;
        auto params = req->get("params");

        auto it = methods_.find(method);
        if (it == methods_.end())
            return R"({"jsonrpc":"2.0","error":{"code":-32601,"message":"Method not found"},"id":)" + id_str + "}";

        auto result = it->second(params
            ? std::shared_ptr<util::JsonValue>(req, const_cast<util::JsonValue*>(params))
            : json_null());
        return R"({"jsonrpc":"2.0","result":)" + stringify(result) + R"(,"id":)" + id_str + "}";

    } catch (const std::exception& e) {
        return R"({"jsonrpc":"2.0","error":{"code":-32603,"message":")" +
               std::string(e.what()) + R"("},"id":null})";
    } catch (...) {
        return R"({"jsonrpc":"2.0","error":{"code":-32603,"message":"internal error"},"id":null})";
    }
}

std::string JsonRpcServer::http_response(int status, const std::string& body) {
    std::ostringstream ss;
    ss << "HTTP/1.1 " << status << " OK\r\n"
       << "Content-Type: application/json\r\n"
       << "Content-Length: " << body.size() << "\r\n"
       << "Connection: close\r\n"
       << "\r\n"
       << body;
    return ss.str();
}

} // namespace khanary::io
