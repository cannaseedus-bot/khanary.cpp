#include "../kuhul/phase_driver_lock.h"

#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: KHANARY_PHASE_DRIVER_LOCK_TEST <repo-root> <alternate-cwd>\n";
        return 2;
    }

    const fs::path repo_root = fs::absolute(argv[1]).lexically_normal();
    const fs::path alternate_cwd = fs::absolute(argv[2]).lexically_normal();
    std::error_code ec;
    fs::current_path(alternate_cwd, ec);
    if (ec) {
        std::cerr << "failed to set alternate cwd: " << alternate_cwd.string()
                  << " (" << ec.message() << ")\n";
        return 3;
    }

    const fs::path fold_runtime = repo_root / "data/schema/fold.runtime.json";
    const fs::path runtime_meta = repo_root / "data/schema/kuhul.fold-runtime.json";

    khanary::kuhul::PhaseDriverLock lock;
    std::string err;
    if (!lock.load(fold_runtime.string(), runtime_meta.string(), &err)) {
        std::cerr << "phase driver lock load failed: " << err << "\n";
        return 1;
    }
    if (!lock.enabled() || !lock.strict() || !lock.ready()) {
        std::cerr << "unexpected phase driver state: " << lock.summary() << "\n";
        return 1;
    }
    if (lock.entries().size() != 6) {
        std::cerr << "expected 6 phase entries, got " << lock.entries().size() << "\n";
        return 1;
    }
    for (const auto& row : lock.entries()) {
        if (!row.exists || !row.hash_declared || !row.hash_match || !row.verified) {
            std::cerr << "unverified phase driver: " << row.phase_name
                      << " path=" << row.dll_path
                      << " error=" << row.error << "\n";
            return 1;
        }
        if (!lock.allow_phase(row.phase, &err)) {
            std::cerr << "phase not allowed: " << row.phase_name
                      << " error=" << err << "\n";
            return 1;
        }
    }

    return 0;
}
