#include "config.h"
#include "manifest/manifest_registry.h"
#include "manifest/manifest_loader.h"
#include "kuhul/phase_engine.h"
#include "kuhul/fold_engine.h"
#include "kuhul/xcfe_executor.h"
#include "kuhul/gram_store.h"
#include "kuhul/kuhul_vm.h"
#include "gpu/d3d11_device.h"
#include "gpu/dml_ops.h"
#include "gpu/dds_loader.h"
#include "gpu/hot_swap.h"
#include "inference/tokenizer.h"
#include "inference/inference_engine.h"
#include "micronaut/micronaut_registry.h"
#include "micronaut/track_loader.h"
#include "micronaut/aiml_engine.h"
#include "micronaut/factory.h"
#include "micronaut/sandbox_inspector.h"
#include "world/world_state.h"
#include "world/field_dag.h"
#include "memory/perception.h"
#include "memory/confidence.h"
#include "memory/semantic_graph.h"
#include "memory/scoring.h"
#include "io/json_rpc_server.h"
#include "io/mcp_server.h"
#include "io/jrom.h"
#include "io/request_router.h"
#include "util/json.h"

#include <iostream>
#include <string>
#include <csignal>
#include <atomic>
#include <thread>
#include <filesystem>

namespace fs = std::filesystem;
using namespace khanary;

static std::atomic<bool> g_shutdown{false};

static void signal_handler(int) { g_shutdown = true; }

struct AppConfig {
    uint16_t    port_mcp  = PORT_MCP;
    uint16_t    port_rpc  = PORT_JSON_RPC;
    std::string data_dir  = "data";
    std::string weights_dir;
    std::string gpu_hint  = "Intel";
    bool        verbose   = false;
};

static AppConfig parse_args(int argc, char* argv[]) {
    AppConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string {
            return (i + 1 < argc) ? argv[++i] : "";
        };
        if (arg == "--port-mcp")      cfg.port_mcp     = (uint16_t)std::stoi(next());
        else if (arg == "--port-rpc") cfg.port_rpc      = (uint16_t)std::stoi(next());
        else if (arg == "--data-dir") cfg.data_dir      = next();
        else if (arg == "--weights-dir") cfg.weights_dir = next();
        else if (arg == "--gpu")      cfg.gpu_hint      = next();
        else if (arg == "--verbose")  cfg.verbose       = true;
        else if (arg == "--help") {
            std::cout << "KHANARY.exe\n"
                      << "  --port-mcp <n>      MCP server port (default: 8764)\n"
                      << "  --port-rpc <n>      JSON-RPC server port (default: 8787)\n"
                      << "  --data-dir <path>   Data directory (default: ./data)\n"
                      << "  --weights-dir <path> Weight shards directory\n"
                      << "  --gpu <hint>        GPU adapter hint (default: Intel)\n"
                      << "  --verbose           Verbose logging\n";
            std::exit(0);
        }
    }
    return cfg;
}

int main(int argc, char* argv[]) {
    auto cfg = parse_args(argc, argv);

    std::signal(SIGINT,  signal_handler);
    std::signal(SIGTERM, signal_handler);

    std::cout << "KHANARY starting up\n"
              << "  data:    " << cfg.data_dir  << "\n"
              << "  MCP:     " << cfg.port_mcp  << "\n"
              << "  RPC:     " << cfg.port_rpc  << "\n";

    // ── GPU ───────────────────────────────────────────────────────────────
    gpu::D3D11Device device;
    if (!device.init(cfg.gpu_hint)) {
        std::cerr << "WARN: D3D11 device init failed — GPU features disabled\n";
    } else {
        const auto& ai = device.adapter_info();
        std::cout << "  GPU:     " << ai.description
                  << " (" << (ai.dedicated_video_memory / (1024*1024)) << " MB VRAM)\n";
    }

    gpu::HotSwapManager hot_swap;
    gpu::DMLOps dml_ops(device);
    dml_ops.init();

    // ── Manifest ──────────────────────────────────────────────────────────
    manifest::ManifestRegistry manifest_reg;
    manifest::load_all(manifest_reg, cfg.data_dir);
    std::cout << "  Manifest: " << manifest_reg.size() << " keys loaded\n";

    // ── K'uhul Runtime ────────────────────────────────────────────────────
    kuhul::PhaseEngine   phase_engine;
    kuhul::FoldEngine    fold_engine(phase_engine);
    kuhul::XCFEExecutor  xcfe;
    kuhul::GramStore     gram_store(cfg.data_dir + "/journal/tool-events.jsonl");
    kuhul::KuhulVM       vm(phase_engine, fold_engine, xcfe, gram_store);

    // ── Micronaut Registry ────────────────────────────────────────────────
    micronaut::MicronautRegistry mu_reg;
    micronaut::TrackLoader       track_loader(mu_reg);
    int n_tracks = track_loader.load_dir(cfg.data_dir + "/tracks");
    std::cout << "  Tracks:  " << n_tracks << " semantic tracks loaded\n";

    micronaut::AIMLEngine aiml;
    int n_aiml = aiml.load_dir(cfg.data_dir + "/alice");
    if (n_aiml > 0)
        std::cout << "  AIML:    " << n_aiml << " files loaded ("
                  << aiml.category_count() << " categories)\n";

    micronaut::Factory factory(mu_reg, CONFIDENCE_DELETE_FLOOR);

    // ── Sandbox Inspector ─────────────────────────────────────────────────
    // Populate from the loaded track entries — each track's declared sandbox
    // is now authoritative for runtime enforcement and JSON inspection.
    micronaut::SandboxInspector sandbox_inspector;
    for (const auto& id : mu_reg.ids()) {
        const auto* e = mu_reg.get(id);
        if (e) sandbox_inspector.register_track(id, e->sandbox);
    }
    std::cout << "  Sandbox: " << mu_reg.ids().size() << " track sandboxes registered\n";

    // Wire sandbox enforcement into the VM — gramRead/gramWrite now gate-checked
    vm.set_sandbox(&sandbox_inspector);

    // ── K-CUBE World State ────────────────────────────────────────────────────
    world::WorldState world_state(mu_reg, sandbox_inspector);
    world_state.rebuild();
    std::cout << "  World:   " << world_state.nodes().size() << " track nodes in K-CUBE\n";

    // ── Startup projection — write field.xml (K-CUBE semantic field DAG) ────
    {
        world::FieldDag field_dag(world_state, mu_reg);
        field_dag.build();
        auto world_dir = cfg.data_dir + "/world";
        fs::create_directories(world_dir);
        std::ofstream fx(world_dir + "/field.xml");
        if (fx) fx << field_dag.to_xml();

        std::ofstream fs(world_dir + "/field.svg");
        if (fs) fs << field_dag.to_svg();

        std::cout << "  Field:   " << world_dir << "/field.{xml,svg} ("
                  << field_dag.nodes().size() << " nodes, "
                  << field_dag.edges().size() << " edges)\n";
    }

    // ── Memory Stores ─────────────────────────────────────────────────────
    memory::PerceptionStore perception;
    memory::ConfidenceStore confidence(cfg.data_dir + "/confidence.json");
    memory::SemanticGraph   sem_graph;
    memory::MicronautScorer scorer(perception, confidence, sem_graph);

    // ── Load DDS weight shards ────────────────────────────────────────────
    if (!cfg.weights_dir.empty() && fs::is_directory(cfg.weights_dir)) {
        gpu::DDSLoader dds_loader(device);
        for (const auto& entry : fs::directory_iterator(cfg.weights_dir)) {
            if (entry.path().extension() == ".scxqdds") {
                try {
                    auto shards = dds_loader.load(entry.path().string());
                    for (auto& s : shards) {
                        if (s.crc_ok && s.buffer)
                            hot_swap.make_resident(s.name,
                                std::shared_ptr<gpu::TensorBuffer>(std::move(s.buffer)));
                    }
                    if (cfg.verbose)
                        std::cout << "  Loaded: " << entry.path().filename() << "\n";
                } catch (const std::exception& e) {
                    std::cerr << "  WARN: " << e.what() << "\n";
                }
            }
        }
        std::cout << "  Resident: "
                  << (hot_swap.resident_bytes() / (1024*1024)) << " MiB / "
                  << (hot_swap.ceiling() / (1024*1024)) << " MiB\n";
    }

    // ── JROM Journal ─────────────────────────────────────────────────────
    io::JROM jrom(cfg.data_dir + "/journal/tool-events.jsonl");

    // ── Servers ───────────────────────────────────────────────────────────
    io::MCPServer mcp_server(
        cfg.port_mcp, manifest_reg, vm, mu_reg, factory,
        sandbox_inspector, confidence, world_state, jrom);

    io::JsonRpcServer rpc_server(cfg.port_rpc);
    io::RequestRouter router(mu_reg, scorer);

    // Wire JSON-RPC routes
    rpc_server.register_method("resolve_manifest", [&](const util::JsonPtr& p) {
        auto key = p ? p->get_string("key") : std::nullopt;
        if (!key) return util::json_null();
        auto expanded = manifest_reg.resolve(*key);
        util::JsonArray arr;
        for (const auto& s : expanded) arr.push_back(util::json_string(s));
        return util::json_array(std::move(arr));
    });

    rpc_server.register_method("fold_query", [&](const util::JsonPtr& p) {
        auto query = p ? p->get_string("query") : std::nullopt;
        if (!query) return util::json_null();
        io::RouteRequest req;
        req.query = *query;
        req.phase = kuhul::Phase::Pop;
        auto result = router.route(req);
        util::JsonObject obj;
        obj["micronaut"] = util::json_string(result.micronaut_id);
        obj["score"]     = util::json_number(result.score);
        // LBS blend: [{track, weight}, ...] normalized bone weights
        util::JsonArray blend_arr;
        for (const auto& [tid, w] : result.blend) {
            util::JsonObject bone;
            bone["track"]  = util::json_string(tid);
            bone["weight"] = util::json_number(static_cast<double>(w));
            blend_arr.push_back(util::json_object(std::move(bone)));
        }
        obj["blend"] = util::json_array(std::move(blend_arr));
        return util::json_object(std::move(obj));
    });

    rpc_server.register_method("micronaut_dispatch", [&](const util::JsonPtr& p) {
        auto query = p ? p->get_string("query") : std::nullopt;
        if (!query) return util::json_null();
        io::RouteRequest req;
        req.query = *query;
        req.phase = kuhul::Phase::Wo;
        auto result = router.route(req);
        return result.response ? result.response : util::json_null();
    });

    // Start servers
    if (!mcp_server.start())
        std::cerr << "WARN: MCP server failed to start on port " << cfg.port_mcp << "\n";
    else
        std::cout << "  MCP:     listening on :" << cfg.port_mcp << "\n";

    if (!rpc_server.start())
        std::cerr << "WARN: JSON-RPC server failed to start on port " << cfg.port_rpc << "\n";
    else
        std::cout << "  RPC:     listening on :" << cfg.port_rpc << "\n";

    jrom.log("startup", "{\"ports\":{\"mcp\":" + std::to_string(cfg.port_mcp) +
             ",\"rpc\":" + std::to_string(cfg.port_rpc) + "}}");

    std::cout << "\nKHANARY ready. Press Ctrl+C to stop.\n";

    // Main loop
    while (!g_shutdown) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cout << "\nShutting down...\n";
    mcp_server.stop();
    rpc_server.stop();
    confidence.save(cfg.data_dir + "/confidence.json");
    gram_store.gramCheckpoint();
    jrom.log("shutdown", "{}");

    return 0;
}
