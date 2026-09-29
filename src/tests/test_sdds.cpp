#include "conformance.h"
#include "../gpu/sdds_format.h"
#include <filesystem>
#include <fstream>

using namespace khanary;
using Bytes = std::vector<uint8_t>;
static void word(Bytes& b, uint32_t n) {
    for (int i = 0; i < 4; ++i) b.push_back(static_cast<uint8_t>(n >> (8 * i)));
}
static Bytes chunk(const std::string& name, uint32_t format = 0) {
    Bytes b(64, 0);
    std::copy(name.begin(), name.end(), b.begin());
    Bytes payload(format == 2 ? 2 : 4, 0);
    word(b, 1); word(b, format); word(b, static_cast<uint32_t>(payload.size()));
    word(b, gpu::sdds_crc32(payload.data(), payload.size()));
    b.insert(b.end(), payload.begin(), payload.end());
    return b;
}
static Bytes container(uint32_t format = 0) {
    Bytes b{'S','D','D','S'}; word(b, 1); word(b, 2);
    for (const char* name : {"first", "last"}) {
        const auto c = chunk(name, format); b.insert(b.end(), c.begin(), c.end());
    }
    return b;
}
int main() {
    return test::run("sdds", [](test::Suite& suite) {
        const auto path = std::filesystem::current_path() / "conformance-sdds.bin";
        auto save = [&](const Bytes& bytes) {
            std::ofstream out(path, std::ios::binary);
            out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
            if (!out) throw std::runtime_error("cannot write SDDS fixture");
        };
        const std::string crc_input = "123456789";
        suite.check("crc32_golden", gpu::sdds_crc32(reinterpret_cast<const uint8_t*>(crc_input.data()), crc_input.size()) == 0xcbf43926u);
        for (uint32_t format = 0; format < 3; ++format) {
            save(container(format));
            suite.check("valid_format_" + std::to_string(format), gpu::validate_sdds(path.string()) && gpu::read_sdds(path.string()).size() == 2);
        }
        const auto valid = container();
        bool all_truncations_rejected = true;
        for (size_t size = 0; size < valid.size(); ++size) {
            save(Bytes(valid.begin(), valid.begin() + static_cast<ptrdiff_t>(size)));
            if (gpu::validate_sdds(path.string())) all_truncations_rejected = false;
        }
        suite.check("every_truncation_rejected", all_truncations_rejected);
        auto invalid = [&](const std::string& name, Bytes bytes) {
            save(bytes);
            suite.check(name + "/validator", !gpu::validate_sdds(path.string()));
            suite.rejects(name + "/reader", [&] { (void)gpu::read_sdds(path.string()); });
        };
        auto b = valid; b[4] = 2; invalid("version", b);
        b = valid; b.back() ^= 1; invalid("crc", b);
        b = valid; b[80] = 3; invalid("format", b);
        b = valid; b[84] = 3; invalid("size", b);
        b = valid; b.push_back(0); invalid("trailing", b);
        b = valid; std::fill(b.begin() + 12, b.begin() + 76, 'a'); invalid("name", b);
        b = valid; std::copy(b.begin() + 12, b.begin() + 76, b.begin() + 96); invalid("duplicate", b);
        b = valid; b[8] = 0; invalid("empty_count", b);
        b = valid; b[0] = 'X'; invalid("magic", b);
    }, 23);
}
