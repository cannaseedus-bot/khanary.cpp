#include "stb_map.h"
#include "sdds_format.h"  // sdds_crc32
#include <fstream>
#include <stdexcept>
#include <cstring>

namespace khanary::gpu {

static_assert(sizeof(STBFieldRecord)    == 36, "STBFieldRecord layout changed");
static_assert(sizeof(STBFoldRecord)     == 20, "STBFoldRecord layout changed");
static_assert(sizeof(STBGramRecord)     == 60, "STBGramRecord layout changed");
static_assert(sizeof(STBGeodesicRecord) == 36, "STBGeodesicRecord layout changed");

// ── Internal helpers ──────────────────────────────────────────────────────

static void write_u32(std::ofstream& f, uint32_t v) {
    const uint8_t b[4] = {
        static_cast<uint8_t>(v),
        static_cast<uint8_t>(v >>  8),
        static_cast<uint8_t>(v >> 16),
        static_cast<uint8_t>(v >> 24)
    };
    f.write(reinterpret_cast<const char*>(b), 4);
}

// ── Builders ──────────────────────────────────────────────────────────────

template<class Record>
static STBSection make_section(STBSectionType type, const std::vector<Record>& recs) {
    STBSection sec;
    sec.type         = type;
    sec.record_count = static_cast<uint32_t>(recs.size());
    sec.data.resize(recs.size() * sizeof(Record));
    for (size_t i = 0; i < recs.size(); ++i)
        std::memcpy(sec.data.data() + i * sizeof(Record), &recs[i], sizeof(Record));
    return sec;
}

STBSection make_field_section(const std::vector<STBFieldRecord>& r) {
    return make_section(STBSectionType::Field, r);
}
STBSection make_fold_section(const std::vector<STBFoldRecord>& r) {
    return make_section(STBSectionType::Fold, r);
}
STBSection make_gram_section(const std::vector<STBGramRecord>& r) {
    return make_section(STBSectionType::Gram, r);
}
STBSection make_geodesic_section(const std::vector<STBGeodesicRecord>& r) {
    return make_section(STBSectionType::Geodesic, r);
}

// ── Extractors ────────────────────────────────────────────────────────────

template<class Record>
static std::vector<Record> get_records(const STBSection& sec, STBSectionType expected) {
    if (sec.type != expected) return {};
    if (sec.data.size() % sizeof(Record) != 0) return {};
    std::vector<Record> recs(sec.data.size() / sizeof(Record));
    for (size_t i = 0; i < recs.size(); ++i)
        std::memcpy(&recs[i], sec.data.data() + i * sizeof(Record), sizeof(Record));
    return recs;
}

std::vector<STBFieldRecord>    get_field_records(const STBSection& sec) {
    return get_records<STBFieldRecord>(sec, STBSectionType::Field);
}
std::vector<STBFoldRecord>     get_fold_records(const STBSection& sec) {
    return get_records<STBFoldRecord>(sec, STBSectionType::Fold);
}
std::vector<STBGramRecord>     get_gram_records(const STBSection& sec) {
    return get_records<STBGramRecord>(sec, STBSectionType::Gram);
}
std::vector<STBGeodesicRecord> get_geodesic_records(const STBSection& sec) {
    return get_records<STBGeodesicRecord>(sec, STBSectionType::Geodesic);
}

// ── I/O ──────────────────────────────────────────────────────────────────

void write_stb_map(const STBMap& map, const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("STBMap: cannot open for write: " + path);
    const char magic[4] = {'S', 'T', 'B', 'M'};
    f.write(magic, 4);
    write_u32(f, 1u);
    write_u32(f, static_cast<uint32_t>(map.sections.size()));
    for (const auto& sec : map.sections) {
        write_u32(f, static_cast<uint32_t>(sec.type));
        write_u32(f, sec.record_count);
        write_u32(f, static_cast<uint32_t>(sec.data.size()));
        write_u32(f, sdds_crc32(sec.data.data(), sec.data.size()));
        if (!sec.data.empty())
            f.write(reinterpret_cast<const char*>(sec.data.data()),
                    static_cast<std::streamsize>(sec.data.size()));
    }
    if (!f) throw std::runtime_error("STBMap: write failed: " + path);
}

STBMap read_stb_map(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("STBMap: cannot open: " + path);
    const auto file_size = f.tellg();
    if (file_size < 12) throw std::runtime_error("STBMap: file too small");
    uint64_t remaining = static_cast<uint64_t>(file_size);
    f.seekg(0);

    auto read_bytes = [&](void* dst, size_t n) {
        if (n > remaining || !f.read(static_cast<char*>(dst), static_cast<std::streamsize>(n)))
            throw std::runtime_error("STBMap: truncated");
        remaining -= n;
    };
    auto word = [&]() {
        uint8_t b[4]; read_bytes(b, 4);
        return uint32_t(b[0]) | (uint32_t(b[1]) << 8) | (uint32_t(b[2]) << 16) | (uint32_t(b[3]) << 24);
    };

    char magic[4]; read_bytes(magic, 4);
    if (magic[0] != 'S' || magic[1] != 'T' || magic[2] != 'B' || magic[3] != 'M')
        throw std::runtime_error("STBMap: bad magic");
    if (word() != 1u) throw std::runtime_error("STBMap: unsupported version");
    const uint32_t section_count = word();
    if (section_count > 256) throw std::runtime_error("STBMap: too many sections");

    STBMap map;
    for (uint32_t i = 0; i < section_count; ++i) {
        const uint32_t type_raw    = word();
        const uint32_t rec_count   = word();
        const uint32_t data_bytes  = word();
        const uint32_t stored_crc  = word();
        if (type_raw > 3) throw std::runtime_error("STBMap: unknown section type");
        if (data_bytes > remaining) throw std::runtime_error("STBMap: section overruns file");
        STBSection sec;
        sec.type         = static_cast<STBSectionType>(type_raw);
        sec.record_count = rec_count;
        sec.data.resize(data_bytes);
        read_bytes(sec.data.data(), data_bytes);
        if (sdds_crc32(sec.data.data(), data_bytes) != stored_crc)
            throw std::runtime_error("STBMap: CRC mismatch");
        map.sections.push_back(std::move(sec));
    }
    if (remaining) throw std::runtime_error("STBMap: trailing bytes");
    return map;
}

} // namespace khanary::gpu
