#include "dds_loader.h"
#include <fstream>
#include <stdexcept>

namespace khanary::gpu {

static uint32_t crc32_table[256];
static bool crc32_init = false;

static void init_crc32() {
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t c = i;
        for (int j = 0; j < 8; ++j)
            c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
        crc32_table[i] = c;
    }
    crc32_init = true;
}

uint32_t DDSLoader::compute_crc32(const uint8_t* data, size_t len) {
    if (!crc32_init) init_crc32();
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; ++i)
        crc = crc32_table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    return crc ^ 0xFFFFFFFFu;
}

DDSLoader::DDSLoader(D3D11Device& dev) : dev_(dev) {}

std::vector<LoadedShard> DDSLoader::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("DDSLoader: cannot open " + path);

    DDSFileHeader hdr{};
    f.read((char*)&hdr, sizeof(hdr));
    if (std::strncmp(hdr.magic, "SDDS", 4) != 0)
        throw std::runtime_error("DDSLoader: bad magic in " + path);
    if (hdr.version != 1)
        throw std::runtime_error("DDSLoader: unsupported version " + std::to_string(hdr.version));

    std::vector<LoadedShard> out;
    out.reserve(hdr.num_chunks);

    for (uint32_t i = 0; i < hdr.num_chunks; ++i) {
        DDSChunkHeader ch{};
        f.read((char*)&ch, sizeof(ch));
        if (!f) break;

        std::vector<uint8_t> data(ch.data_size);
        f.read((char*)data.data(), ch.data_size);
        if (!f) break;

        uint32_t actual_crc = compute_crc32(data.data(), data.size());
        bool crc_ok = (actual_crc == ch.crc32);

        TensorFormat fmt = TensorFormat::F32;
        if (ch.format == 1) fmt = TensorFormat::Q8;
        if (ch.format == 2) fmt = TensorFormat::F16;

        TensorDesc desc;
        desc.name       = ch.name;
        desc.format     = fmt;
        desc.elem_count = ch.elem_count;

        auto tb = std::make_unique<TensorBuffer>(dev_, desc);
        if (tb->is_valid() && !data.empty())
            tb->upload(data.data(), data.size());

        LoadedShard shard;
        shard.name   = ch.name;
        shard.buffer = std::move(tb);
        shard.crc_ok = crc_ok;
        out.push_back(std::move(shard));
    }
    return out;
}

bool DDSLoader::validate_crc(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    DDSFileHeader hdr{};
    f.read((char*)&hdr, sizeof(hdr));
    if (std::strncmp(hdr.magic, "SDDS", 4) != 0) return false;
    for (uint32_t i = 0; i < hdr.num_chunks; ++i) {
        DDSChunkHeader ch{};
        f.read((char*)&ch, sizeof(ch));
        if (!f) return false;
        std::vector<uint8_t> data(ch.data_size);
        f.read((char*)data.data(), ch.data_size);
        if (!f) return false;
        if (compute_crc32(data.data(), data.size()) != ch.crc32) return false;
    }
    return true;
}

} // namespace khanary::gpu
