#include "conformance.h"
#include "../gpu/stb_map.h"
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>

using namespace khanary;
using Bytes = std::vector<uint8_t>;

// Write little-endian uint32 into a byte buffer.
static void push_u32(Bytes& b, uint32_t v) {
    for (int i = 0; i < 4; ++i)
        b.push_back(static_cast<uint8_t>(v >> (8 * i)));
}

// Build a minimal valid STBM binary (no sections) from scratch.
static Bytes minimal_stbm() {
    Bytes b{'S', 'T', 'B', 'M'};
    push_u32(b, 1u);   // version
    push_u32(b, 0u);   // section_count = 0
    return b;
}

int main() {
    return test::run("stb_map", [](test::Suite& suite) {
        const auto tmp = std::filesystem::current_path() / "conformance-stbmap.stbm";
        auto save = [&](const Bytes& bytes) {
            std::ofstream out(tmp, std::ios::binary);
            out.write(reinterpret_cast<const char*>(bytes.data()),
                      static_cast<std::streamsize>(bytes.size()));
            if (!out) throw std::runtime_error("cannot write STBMap fixture");
        };

        // ── Empty map round-trip ──────────────────────────────────────────────
        suite.check("empty_map_round_trip", [&]{
            gpu::STBMap m;
            write_stb_map(m, tmp.string());
            const auto m2 = gpu::read_stb_map(tmp.string());
            std::filesystem::remove(tmp);
            return m2.sections.empty();
        }());

        // ── Field section round-trip ──────────────────────────────────────────
        suite.check("field_section_round_trip", [&]{
            gpu::STBFieldRecord r{};
            r.q[0] = 1.0f; r.q[1] = 0.0f; r.q[2] = 0.0f;
            r.k[0] = 0.5f; r.k[1] = 0.5f; r.k[2] = 0.0f;
            r.v[0] = 0.34f; r.v[1] = 0.8f; r.v[2] = 0.9f;
            gpu::STBMap m;
            m.sections.push_back(gpu::make_field_section({r}));
            write_stb_map(m, tmp.string());
            const auto m2 = gpu::read_stb_map(tmp.string());
            std::filesystem::remove(tmp);
            if (m2.sections.size() != 1) return false;
            const auto recs = gpu::get_field_records(m2.sections[0]);
            return recs.size() == 1
                && std::abs(recs[0].q[0] - 1.0f) < 1e-6f
                && std::abs(recs[0].v[0] - 0.34f) < 1e-6f;
        }());

        suite.check("field_section_record_count", [&]{
            gpu::STBMap m;
            std::vector<gpu::STBFieldRecord> recs(5);
            for (int i = 0; i < 5; ++i) recs[static_cast<size_t>(i)].q[0] = static_cast<float>(i);
            m.sections.push_back(gpu::make_field_section(recs));
            write_stb_map(m, tmp.string());
            const auto m2 = gpu::read_stb_map(tmp.string());
            std::filesystem::remove(tmp);
            return m2.sections.size() == 1 && m2.sections[0].record_count == 5;
        }());

        // ── Fold section round-trip ───────────────────────────────────────────
        suite.check("fold_section_round_trip", [&]{
            gpu::STBFoldRecord r{};
            r.from_phase = 1; r.to_phase = 2; r.fold_type = 1;
            r.arc_weight = 0.5f; r.delta_theta = 60.0f;
            gpu::STBMap m;
            m.sections.push_back(gpu::make_fold_section({r}));
            write_stb_map(m, tmp.string());
            const auto m2 = gpu::read_stb_map(tmp.string());
            std::filesystem::remove(tmp);
            if (m2.sections.size() != 1) return false;
            const auto recs = gpu::get_fold_records(m2.sections[0]);
            return recs.size() == 1
                && recs[0].from_phase == 1u
                && recs[0].fold_type  == 1u
                && std::abs(recs[0].delta_theta - 60.0f) < 1e-4f;
        }());

        // ── Gram section round-trip ───────────────────────────────────────────
        suite.check("gram_section_round_trip", [&]{
            gpu::STBGramRecord r{};
            constexpr const char kKey[] = "hello world";
            std::memset(r.key, 0, sizeof(r.key));
            std::memcpy(r.key, kKey, sizeof(kKey) - 1);
            r.q_weight = 1.2f; r.k_weight = 0.8f; r.v_weight = 0.6f;
            gpu::STBMap m;
            m.sections.push_back(gpu::make_gram_section({r}));
            write_stb_map(m, tmp.string());
            const auto m2 = gpu::read_stb_map(tmp.string());
            std::filesystem::remove(tmp);
            if (m2.sections.size() != 1) return false;
            const auto recs = gpu::get_gram_records(m2.sections[0]);
            return recs.size() == 1
                && std::string(recs[0].key) == "hello world"
                && std::abs(recs[0].q_weight - 1.2f) < 1e-5f;
        }());

        // ── Geodesic section round-trip ───────────────────────────────────────
        suite.check("geodesic_section_round_trip", [&]{
            gpu::STBGeodesicRecord r{};
            r.x = 1.0f; r.y = 2.0f; r.z = 3.0f;
            r.local_entropy = 0.7f; r.geodesic_entropy = 0.9f;
            r.unruh_temp = 0.01f; r.divergence = -0.3f;
            r.circumradius_R = 2.0f; r.inradius_r = 0.75f;
            gpu::STBMap m;
            m.sections.push_back(gpu::make_geodesic_section({r}));
            write_stb_map(m, tmp.string());
            const auto m2 = gpu::read_stb_map(tmp.string());
            std::filesystem::remove(tmp);
            if (m2.sections.size() != 1) return false;
            const auto recs = gpu::get_geodesic_records(m2.sections[0]);
            return recs.size() == 1
                && std::abs(recs[0].x - 1.0f) < 1e-6f
                && std::abs(recs[0].divergence - (-0.3f)) < 1e-5f
                && std::abs(recs[0].circumradius_R - 2.0f) < 1e-5f
                && std::abs(recs[0].inradius_r - 0.75f) < 1e-5f;
        }());

        // ── Multi-section map ─────────────────────────────────────────────────
        suite.check("multi_section_round_trip", [&]{
            gpu::STBMap m;
            m.sections.push_back(gpu::make_field_section({gpu::STBFieldRecord{}}));
            m.sections.push_back(gpu::make_fold_section({gpu::STBFoldRecord{}}));
            m.sections.push_back(gpu::make_geodesic_section({gpu::STBGeodesicRecord{}}));
            write_stb_map(m, tmp.string());
            const auto m2 = gpu::read_stb_map(tmp.string());
            std::filesystem::remove(tmp);
            return m2.sections.size() == 3
                && m2.sections[0].type == gpu::STBSectionType::Field
                && m2.sections[1].type == gpu::STBSectionType::Fold
                && m2.sections[2].type == gpu::STBSectionType::Geodesic;
        }());

        // ── Corruption / rejection tests ──────────────────────────────────────
        auto reject = [&](const std::string& label, Bytes bytes) {
            save(bytes);
            suite.rejects(label, [&] { (void)gpu::read_stb_map(tmp.string()); });
            std::filesystem::remove(tmp);
        };

        const auto valid = minimal_stbm();
        Bytes b;

        b = valid; b[0] = 'X';
        reject("bad_magic_throws", b);

        b = valid; b[4] = 2;
        reject("bad_version_throws", b);

        b = valid; b.push_back(0);
        reject("trailing_bytes_throws", b);

        b = Bytes(valid.begin(), valid.begin() + 8);
        reject("truncated_throws", b);

        // CRC corruption: write a single-section map, then flip a data byte
        suite.check("crc_corruption_throws", [&]{
            gpu::STBFieldRecord r{}; r.q[0] = 1.0f;
            gpu::STBMap m;
            m.sections.push_back(gpu::make_field_section({r}));
            write_stb_map(m, tmp.string());
            // Read raw bytes, flip the last data byte
            Bytes raw;
            {
                std::ifstream in(tmp, std::ios::binary);
                raw.assign((std::istreambuf_iterator<char>(in)),
                             std::istreambuf_iterator<char>());
            }
            raw.back() ^= 0xFFu;
            save(raw);
            bool threw = false;
            try { (void)gpu::read_stb_map(tmp.string()); }
            catch (const std::exception&) { threw = true; }
            std::filesystem::remove(tmp);
            return threw;
        }());

    }, 12);
}
