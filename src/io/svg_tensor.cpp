#include "svg_tensor.h"
#include <charconv>
#include <fstream>
#include <string_view>
#include <unordered_map>

namespace khanary::io {
namespace {

static float pf(const std::string& s, float def = 0.0f) {
    if (s.empty()) return def;
    float v = def;
    std::from_chars(s.data(), s.data() + s.size(), v);
    return v;
}

static std::vector<float> parse_float_vec(const std::string& s) {
    std::vector<float> out;
    size_t pos = 0;
    while (pos <= s.size()) {
        const auto comma = s.find(',', pos);
        const auto end   = comma == std::string::npos ? s.size() : comma;
        std::string_view tok{ s.data() + pos, end - pos };
        while (!tok.empty() && tok.front() == ' ') tok.remove_prefix(1);
        while (!tok.empty() && tok.back()  == ' ') tok.remove_suffix(1);
        if (!tok.empty()) {
            float v = 0.0f;
            if (std::from_chars(tok.data(), tok.data() + tok.size(), v).ec == std::errc{})
                out.push_back(v);
        }
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    return out;
}

// Extract k:* attributes from an SVG element's body string (the text between < and >).
// Only attributes where "k:" is preceded by whitespace are extracted (this excludes
// xmlns:k declarations and namespace-prefixed element names like <k:elem>).
static std::unordered_map<std::string, std::string>
extract_k_attrs(const std::string& body) {
    std::unordered_map<std::string, std::string> attrs;
    size_t pos = 0;
    const size_t n = body.size();
    while (pos < n) {
        const auto kp = body.find("k:", pos);
        if (kp == std::string::npos) break;
        // Require whitespace before k: to distinguish attributes from xmlns:k or <k:elem>
        if (kp == 0) { pos = 2; continue; }
        const char prev = body[kp - 1];
        if (prev != ' ' && prev != '\t' && prev != '\n' && prev != '\r') {
            pos = kp + 1;
            continue;
        }
        const size_t name_start = kp + 2;
        const size_t eq = body.find('=', name_start);
        if (eq == std::string::npos) break;
        const std::string name = body.substr(name_start, eq - name_start);
        // Name must not contain whitespace
        bool clean = !name.empty();
        for (char c : name)
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') { clean = false; break; }
        if (!clean) { pos = eq + 1; continue; }
        if (eq + 1 >= n) break;
        const char q = body[eq + 1];
        if (q != '"' && q != '\'') { pos = eq + 1; continue; }
        const size_t vs = eq + 2;
        const size_t ve = body.find(q, vs);
        if (ve == std::string::npos) break;
        attrs[name] = body.substr(vs, ve - vs);
        pos = ve + 1;
    }
    return attrs;
}

// Walk the SVG text and collect attribute maps for elements that carry k:* attributes.
static std::vector<std::unordered_map<std::string, std::string>>
scan_elements(const std::string& text) {
    std::vector<std::unordered_map<std::string, std::string>> result;
    size_t pos = 0;
    const size_t n = text.size();
    while (pos < n) {
        const auto lt = text.find('<', pos);
        if (lt == std::string::npos) break;
        // Skip XML comment <!-- ... -->
        if (n - lt > 3
            && text[lt+1] == '!' && text[lt+2] == '-' && text[lt+3] == '-') {
            const auto end = text.find("-->", lt + 4);
            pos = (end == std::string::npos) ? n : end + 3;
            continue;
        }
        // Skip CDATA <![CDATA[...]]>
        if (n - lt > 8 && text.compare(lt, 9, "<![CDATA[") == 0) {
            const auto end = text.find("]]>", lt + 9);
            pos = (end == std::string::npos) ? n : end + 3;
            continue;
        }
        // Skip processing instruction <?...?>
        if (lt + 1 < n && text[lt+1] == '?') {
            const auto end = text.find("?>", lt + 2);
            pos = (end == std::string::npos) ? n : end + 2;
            continue;
        }
        // Skip closing tag </...>
        if (lt + 1 < n && text[lt+1] == '/') {
            const auto end = text.find('>', lt + 2);
            pos = (end == std::string::npos) ? n : end + 1;
            continue;
        }
        // Find end of tag, honouring quoted attribute values
        size_t te = lt + 1;
        bool in_q = false; char qc = 0;
        while (te < n) {
            const char c = text[te];
            if (in_q) { if (c == qc) in_q = false; }
            else {
                if (c == '"' || c == '\'') { in_q = true; qc = c; }
                else if (c == '>') break;
            }
            ++te;
        }
        if (te >= n) break;
        const std::string body = text.substr(lt + 1, te - lt - 1);
        if (body.find("k:") != std::string::npos) {
            auto attrs = extract_k_attrs(body);
            if (!attrs.empty()) result.push_back(std::move(attrs));
        }
        pos = te + 1;
    }
    return result;
}

static std::unique_ptr<SVGTensorGraph>
build_graph(
    const std::vector<std::unordered_map<std::string, std::string>>& elements)
{
    auto g = std::make_unique<SVGTensorGraph>();
    for (const auto& a : elements) {
        auto get = [&](const char* key) -> std::string {
            const auto it = a.find(key);
            return it != a.end() ? it->second : std::string{};
        };
        const bool is_edge = a.count("edge_from") != 0 && a.count("edge_to") != 0;
        if (is_edge) {
            SVGTensorEdge e;
            e.from        = get("edge_from");
            e.to          = get("edge_to");
            e.verb        = get("verb");
            e.fold_type   = get("fold_type");
            e.weight      = pf(get("weight"), 1.0f);
            e.delta_theta = pf(get("delta_theta"), 0.0f);
            g->edges.push_back(std::move(e));
        } else if (a.count("track_id") != 0 || a.count("kind") != 0) {
            SVGTensorNode nd;
            nd.track_id   = get("track_id");
            nd.phase_name = get("phase");
            nd.kind       = get("kind");
            if (nd.kind.empty()) nd.kind = "track";
            nd.x          = pf(get("x"));
            nd.y          = pf(get("y"));
            nd.z          = pf(get("z"));
            nd.entropy    = pf(get("entropy"));
            nd.geodesic_entropy = pf(get("geodesic_entropy"), nd.entropy);
            nd.unruh_temp       = pf(get("unruh_temp"));
            nd.divergence       = pf(get("divergence"));
            nd.circumradius_R   = pf(get("circumradius_R"));
            nd.inradius_r       = pf(get("inradius_r"));
            nd.weight     = pf(get("weight"), 1.0f);
            nd.confidence = pf(get("confidence"), 1.0f);
            const auto qv = get("q_vec");
            const auto kv = get("k_vec");
            const auto vv = get("v_vec");
            if (!qv.empty()) nd.q_vec = parse_float_vec(qv);
            if (!kv.empty()) nd.k_vec = parse_float_vec(kv);
            if (!vv.empty()) nd.v_vec = parse_float_vec(vv);
            g->nodes.push_back(std::move(nd));
        }
    }
    return g;
}

} // anonymous namespace

std::unique_ptr<SVGTensorGraph> parse_svg_tensor(const std::string& text) {
    return build_graph(scan_elements(text));
}

std::unique_ptr<SVGTensorGraph> load_svg_tensor(const std::string& path) {
    std::ifstream f(path);
    if (!f) return nullptr;
    const std::string text(
        (std::istreambuf_iterator<char>(f)),
         std::istreambuf_iterator<char>());
    return parse_svg_tensor(text);
}

} // namespace khanary::io
