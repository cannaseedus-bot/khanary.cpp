#include "sdds_format.h"
#include <array>
#include <algorithm>
#include <fstream>
#include <set>
#include <stdexcept>

namespace khanary::gpu {
uint32_t sdds_crc32(const uint8_t* data, size_t size) {
    static const auto table = [] {
        std::array<uint32_t, 256> values{};
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int b = 0; b < 8; ++b) c = (c & 1) ? (0xedb88320u ^ (c >> 1)) : (c >> 1);
            values[i] = c;
        }
        return values;
    }();
    uint32_t crc = 0xffffffffu;
    for (size_t i = 0; i < size; ++i) crc = table[(crc ^ data[i]) & 255] ^ (crc >> 8);
    return crc ^ 0xffffffffu;
}
std::vector<SDDSChunk> read_sdds(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("SDDS: cannot open file");
    const auto size = f.tellg();
    constexpr uint64_t max_bytes = 1879048192ULL;
    if (size < 12 || static_cast<uint64_t>(size) > max_bytes)
        throw std::runtime_error("SDDS: invalid file size");
    uint64_t remaining = static_cast<uint64_t>(size);
    f.seekg(0);
    auto read = [&](void* dest, size_t bytes) {
        if (bytes > remaining || !f.read(static_cast<char*>(dest), static_cast<std::streamsize>(bytes)))
            throw std::runtime_error("SDDS: truncated container");
        remaining -= bytes;
    };
    auto word = [&]() {
        uint8_t b[4]; read(b, 4);
        return uint32_t(b[0]) | (uint32_t(b[1]) << 8) | (uint32_t(b[2]) << 16) | (uint32_t(b[3]) << 24);
    };
    char magic[4]; read(magic, 4);
    if (std::string(magic, 4) != "SDDS") throw std::runtime_error("SDDS: bad magic");
    if (word() != 1) throw std::runtime_error("SDDS: unsupported version");
    const uint32_t count = word();
    if (count == 0 || count > remaining / 80 || count > 65536)
        throw std::runtime_error("SDDS: invalid chunk count");
    std::vector<SDDSChunk> chunks;
    std::set<std::string> names;
    for (uint32_t i = 0; i < count; ++i) {
        char name[64]; read(name, 64);
        const auto end = std::find(name, name + 64, '\0');
        if (end == name || end == name + 64 || std::any_of(end, name + 64, [](char c) { return c != 0; }))
            throw std::runtime_error("SDDS: invalid null-padded name");
        SDDSChunk chunk;
        chunk.name.assign(name, end);
        if (!names.insert(chunk.name).second) throw std::runtime_error("SDDS: duplicate chunk name");
        chunk.elem_count = word();
        chunk.format = word();
        const uint32_t bytes = word(), crc = word();
        if (chunk.format > 2 || chunk.elem_count == 0) throw std::runtime_error("SDDS: invalid tensor descriptor");
        const uint64_t expected = uint64_t(chunk.elem_count) * (chunk.format == 2 ? 2 : 4);
        if (expected != bytes || expected > remaining) throw std::runtime_error("SDDS: inconsistent tensor size");
        chunk.data.resize(bytes);
        read(chunk.data.data(), bytes);
        if (sdds_crc32(chunk.data.data(), bytes) != crc) throw std::runtime_error("SDDS: CRC mismatch");
        chunks.push_back(std::move(chunk));
    }
    if (remaining) throw std::runtime_error("SDDS: trailing bytes");
    return chunks;
}
bool validate_sdds(const std::string& path) {
    try { (void)read_sdds(path); return true; }
    catch (const std::exception&) { return false; }
}
}
