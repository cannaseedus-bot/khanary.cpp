#include "gml_parser.h"
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <optional>

namespace khanary::world {

namespace {

// ── Lexer ─────────────────────────────────────────────────────────────────────

enum class TokKind { IDENT, INT, FLOAT, STRING, LBRACKET, RBRACKET, END };

struct Token {
    TokKind kind = TokKind::END;
    std::string str;
    int64_t ival = 0;
    double  fval = 0.0;
};

struct Lexer {
    const char* p;
    const char* end;

    explicit Lexer(const std::string& s) noexcept
        : p(s.data()), end(s.data() + s.size()) {}

    void skip_ws() noexcept {
        while (p < end && std::isspace(static_cast<unsigned char>(*p)))
            ++p;
    }

    Token next() {
        skip_ws();
        if (p >= end) return {};

        if (*p == '[') { ++p; return {TokKind::LBRACKET, "["}; }
        if (*p == ']') { ++p; return {TokKind::RBRACKET, "]"}; }

        if (*p == '"') {
            ++p;
            std::string s;
            while (p < end && *p != '"') {
                if (*p == '\\' && p + 1 < end) {
                    ++p;
                    switch (*p) {
                        case '"':  s += '"';  break;
                        case '\\': s += '\\'; break;
                        case 'n':  s += '\n'; break;
                        case 't':  s += '\t'; break;
                        default:   s += '\\'; s += *p; break;
                    }
                } else {
                    s += *p;
                }
                ++p;
            }
            if (p < end) ++p; // consume closing "
            return {TokKind::STRING, s};
        }

        // Number: optional sign then digits, optional fractional/exponent part
        const bool sign = (*p == '-' || *p == '+');
        if (sign && p + 1 < end && std::isdigit(static_cast<unsigned char>(*(p + 1)))) {
            return scan_number();
        }
        if (std::isdigit(static_cast<unsigned char>(*p))) {
            return scan_number();
        }

        // Identifier
        if (std::isalpha(static_cast<unsigned char>(*p)) || *p == '_') {
            const char* start = p;
            while (p < end &&
                   (std::isalnum(static_cast<unsigned char>(*p)) || *p == '_'))
                ++p;
            return {TokKind::IDENT, std::string(start, p)};
        }

        // Unknown character — skip and recurse
        ++p;
        return next();
    }

    Token peek() {
        const char* saved = p;
        Token t = next();
        p = saved;
        return t;
    }

private:
    Token scan_number() {
        const char* start = p;
        bool is_float = false;
        if (*p == '-' || *p == '+') ++p;
        while (p < end && std::isdigit(static_cast<unsigned char>(*p))) ++p;
        if (p < end && *p == '.') {
            is_float = true;
            ++p;
            while (p < end && std::isdigit(static_cast<unsigned char>(*p))) ++p;
        }
        if (p < end && (*p == 'e' || *p == 'E')) {
            is_float = true;
            ++p;
            if (p < end && (*p == '+' || *p == '-')) ++p;
            while (p < end && std::isdigit(static_cast<unsigned char>(*p))) ++p;
        }
        std::string ns(start, p);
        if (is_float) {
            Token t{TokKind::FLOAT, ns};
            t.fval = std::stod(ns);
            return t;
        }
        Token t{TokKind::INT, ns};
        t.ival = std::stoll(ns);
        return t;
    }
};

// ── Raw block tree ────────────────────────────────────────────────────────────

struct RawBlock {
    std::string key;
    bool is_block = false;
    int64_t ival  = 0;
    double  fval  = 0.0;
    std::string sval;
    std::vector<RawBlock> children;
};

std::vector<RawBlock> parse_body(Lexer& lex) {
    std::vector<RawBlock> items;
    while (true) {
        Token t = lex.peek();
        if (t.kind == TokKind::RBRACKET || t.kind == TokKind::END) {
            if (t.kind == TokKind::RBRACKET) lex.next();
            break;
        }
        if (t.kind != TokKind::IDENT) { lex.next(); continue; }

        RawBlock rb;
        rb.key = lex.next().str;

        Token val = lex.peek();
        if (val.kind == TokKind::LBRACKET) {
            lex.next();
            rb.is_block  = true;
            rb.children  = parse_body(lex);
        } else if (val.kind == TokKind::INT) {
            lex.next();
            rb.ival = val.ival;
        } else if (val.kind == TokKind::FLOAT) {
            lex.next();
            rb.fval = val.fval;
        } else if (val.kind == TokKind::STRING) {
            lex.next();
            rb.sval = val.str;
        }
        // If val is anything else (IDENT treated as key), leave defaults and
        // don't consume — the next iteration picks it up as a key.
        items.push_back(std::move(rb));
    }
    return items;
}

// ── Typed accessors ───────────────────────────────────────────────────────────

static std::optional<std::string>
get_str(const std::vector<RawBlock>& v, const std::string& k) {
    for (const auto& rb : v)
        if (rb.key == k && !rb.is_block) return rb.sval;
    return std::nullopt;
}

static std::optional<int64_t>
get_int(const std::vector<RawBlock>& v, const std::string& k) {
    for (const auto& rb : v)
        if (rb.key == k && !rb.is_block) return rb.ival;
    return std::nullopt;
}

static std::optional<double>
get_dbl(const std::vector<RawBlock>& v, const std::string& k) {
    for (const auto& rb : v) {
        if (rb.key == k && !rb.is_block) {
            return (rb.fval != 0.0) ? rb.fval : static_cast<double>(rb.ival);
        }
    }
    return std::nullopt;
}

// ── GML object builders ───────────────────────────────────────────────────────

static GmlNode build_node(const std::vector<RawBlock>& items) {
    GmlNode n;
    if (auto v = get_int(items, "id"))    n.id    = *v;
    if (auto v = get_str(items, "label")) n.label = *v;
    if (auto v = get_dbl(items, "x")) { n.x = *v; n.has_x = true; }
    if (auto v = get_dbl(items, "y")) { n.y = *v; n.has_y = true; }
    if (auto v = get_dbl(items, "z")) { n.z = *v; n.has_z = true; }
    if (auto v = get_dbl(items, "confidence")) n.confidence = *v;
    if (auto v = get_dbl(items, "static_weight")) n.static_weight = *v;
    if (auto v = get_dbl(items, "health")) n.health = *v;
    if (auto v = get_dbl(items, "prior_mass")) { n.prior_mass = *v; n.has_prior_mass = true; }
    if (auto v = get_dbl(items, "risk_bound")) { n.risk_bound = *v; n.has_risk_bound = true; }
    if (auto v = get_int(items, "base_version")) { n.base_version = *v; n.has_base_version = true; }
    if (auto v = get_str(items, "mutation")) n.mutation = *v;
    if (auto v = get_dbl(items, "divergence")) { n.divergence = *v; n.has_divergence = true; }
    if (auto v = get_str(items, "singularity")) n.singularity = *v;
    if (auto v = get_dbl(items, "local_entropy"))    { n.local_entropy    = *v; n.has_local_entropy    = true; }
    if (auto v = get_dbl(items, "unruh_temp"))       { n.unruh_temp       = *v; n.has_unruh_temp       = true; }
    if (auto v = get_dbl(items, "bit_density"))      { n.bit_density      = *v; n.has_bit_density      = true; }
    if (auto v = get_dbl(items, "geodesic_entropy")) { n.geodesic_entropy = *v; n.has_geodesic_entropy = true; }
    if (auto v = get_dbl(items, "triangular_scale")) { n.triangular_scale = *v; n.has_triangular_scale = true; }
    if (auto v = get_dbl(items, "circumradius_R"))   { n.circumradius_R   = *v; n.has_circumradius_R   = true; }
    if (auto v = get_dbl(items, "inradius_r"))       { n.inradius_r       = *v; n.has_inradius_r       = true; }
    if (auto v = get_str(items, "boundary_type")) n.boundary_type = *v;
    if (auto v = get_dbl(items, "vertex_pot"))    { n.vertex_pot  = *v; n.has_vertex_pot  = true; }
    if (auto v = get_dbl(items, "phase_shift"))   { n.phase_shift = *v; n.has_phase_shift = true; }
    if (auto v = get_str(items, "text_boundary"))  n.text_boundary  = *v;
    if (auto v = get_str(items, "boundary_token")) n.boundary_token = *v;
    if (auto v = get_str(items, "curve_id"))  n.curve_id = *v;
    if (auto v = get_dbl(items, "parameter_t"))      { n.parameter_t      = *v; n.has_parameter_t      = true; }
    if (auto v = get_dbl(items, "arc_length"))       { n.arc_length       = *v; n.has_arc_length       = true; }
    if (auto v = get_dbl(items, "curvature"))        { n.curvature        = *v; n.has_curvature        = true; }
    if (auto v = get_dbl(items, "scalar_potential")) { n.scalar_potential = *v; n.has_scalar_potential = true; }
    if (auto v = get_str(items, "provider_id"))  n.provider_id  = *v;
    if (auto v = get_str(items, "parameter_id")) n.parameter_id = *v;
    if (auto v = get_int(items, "parameter_generation")) { n.parameter_generation = *v; n.has_parameter_generation = true; }
    if (auto v = get_int(items, "training_step"))        { n.training_step        = *v; n.has_training_step        = true; }
    if (auto v = get_dbl(items, "loss_value"))    { n.loss_value    = *v; n.has_loss_value    = true; }
    if (auto v = get_dbl(items, "grad_norm"))     { n.grad_norm     = *v; n.has_grad_norm     = true; }
    if (auto v = get_dbl(items, "learning_rate")) { n.learning_rate = *v; n.has_learning_rate = true; }
    if (auto v = get_dbl(items, "gradient_clip")) { n.gradient_clip = *v; n.has_gradient_clip = true; }
    if (auto v = get_dbl(items, "entropy_spike"))      { n.entropy_spike      = *v; n.has_entropy_spike      = true; }
    if (auto v = get_dbl(items, "entropy_bound"))      { n.entropy_bound      = *v; n.has_entropy_bound      = true; }
    if (auto v = get_dbl(items, "diffusion_beta_inv")) { n.diffusion_beta_inv = *v; n.has_diffusion_beta_inv = true; }
    if (auto v = get_int(items, "tensor_rank_p"))       { n.tensor_rank_p       = *v; n.has_tensor_rank_p       = true; }
    if (auto v = get_int(items, "tensor_rank_q"))       { n.tensor_rank_q       = *v; n.has_tensor_rank_q       = true; }
    if (auto v = get_int(items, "is_smooth_section"))   { n.is_smooth_section   = *v; n.has_is_smooth_section   = true; }
    if (auto v = get_dbl(items, "riemann_curvature_scalar")) { n.riemann_curvature_scalar = *v; n.has_riemann_curvature_scalar = true; }
    if (auto v = get_dbl(items, "jacobian_determinant")) { n.jacobian_determinant = *v; n.has_jacobian_determinant = true; }
    return n;
}

static GmlEdge build_edge(const std::vector<RawBlock>& items) {
    GmlEdge e;
    if (auto v = get_int(items, "id")) { e.id = *v; e.has_id = true; }
    if (auto v = get_int(items, "source")) e.source = *v;
    if (auto v = get_int(items, "target")) e.target = *v;
    if (auto v = get_str(items, "label"))  e.label  = *v;
    if (auto v = get_str(items, "verb"))   e.verb   = *v;
    if (e.verb.empty()) {
        if (auto v = get_str(items, "relation")) e.verb = *v;
    }
    if (e.label.empty() && !e.verb.empty()) e.label = e.verb;
    if (auto v = get_dbl(items, "weight")) { e.weight = *v; e.has_weight = true; }
    if (auto v = get_int(items, "dual_twin")) { e.dual_twin = *v; e.has_dual_twin = true; }
    if (auto v = get_dbl(items, "loss_cost")) { e.loss_cost = *v; e.has_loss_cost = true; }
    if (auto v = get_str(items, "mutation")) e.mutation = *v;
    if (auto v = get_dbl(items, "vx")) { e.vx = *v; e.has_vx = true; }
    if (auto v = get_dbl(items, "vy")) { e.vy = *v; e.has_vy = true; }
    if (auto v = get_dbl(items, "vz")) { e.vz = *v; e.has_vz = true; }
    if (auto v = get_dbl(items, "eh_x")) { e.eh_x = *v; e.has_eh_x = true; }
    if (auto v = get_dbl(items, "eh_y")) { e.eh_y = *v; e.has_eh_y = true; }
    if (auto v = get_dbl(items, "eh_z")) { e.eh_z = *v; e.has_eh_z = true; }
    if (auto v = get_dbl(items, "length"))       { e.length       = *v; e.has_length       = true; }
    if (auto v = get_dbl(items, "potential"))    { e.potential    = *v; e.has_potential    = true; }
    if (auto v = get_dbl(items, "refractive"))   { e.refractive   = *v; e.has_refractive   = true; }
    if (auto v = get_dbl(items, "wave_number"))  { e.wave_number  = *v; e.has_wave_number  = true; }
    if (auto v = get_dbl(items, "frequency"))    { e.frequency    = *v; e.has_frequency    = true; }
    if (auto v = get_dbl(items, "amplitude_re")) { e.amplitude_re = *v; e.has_amplitude_re = true; }
    if (auto v = get_dbl(items, "amplitude_im")) { e.amplitude_im = *v; e.has_amplitude_im = true; }
    if (auto v = get_str(items, "curve_id"))   e.curve_id   = *v;
    if (auto v = get_dbl(items, "t0"))           { e.t0           = *v; e.has_t0           = true; }
    if (auto v = get_dbl(items, "t1"))           { e.t1           = *v; e.has_t1           = true; }
    if (auto v = get_str(items, "field_kind")) e.field_kind = *v;
    if (auto v = get_dbl(items, "curl_x"))       { e.curl_x       = *v; e.has_curl_x       = true; }
    if (auto v = get_dbl(items, "curl_y"))       { e.curl_y       = *v; e.has_curl_y       = true; }
    if (auto v = get_dbl(items, "curl_z"))       { e.curl_z       = *v; e.has_curl_z       = true; }
    if (auto v = get_dbl(items, "potential_delta")) { e.potential_delta = *v; e.has_potential_delta = true; }
    if (auto v = get_str(items, "provider_id"))  e.provider_id  = *v;
    if (auto v = get_str(items, "parameter_id")) e.parameter_id = *v;
    if (auto v = get_int(items, "training_step")) { e.training_step = *v; e.has_training_step = true; }
    if (auto v = get_dbl(items, "loss_value"))    { e.loss_value    = *v; e.has_loss_value    = true; }
    if (auto v = get_dbl(items, "grad_norm"))     { e.grad_norm     = *v; e.has_grad_norm     = true; }
    if (auto v = get_dbl(items, "learning_rate")) { e.learning_rate = *v; e.has_learning_rate = true; }
    if (auto v = get_dbl(items, "gradient_clip")) { e.gradient_clip = *v; e.has_gradient_clip = true; }
    if (auto v = get_dbl(items, "entropy_delta"))      { e.entropy_delta      = *v; e.has_entropy_delta      = true; }
    if (auto v = get_dbl(items, "diffusion_beta_inv")) { e.diffusion_beta_inv = *v; e.has_diffusion_beta_inv = true; }
    if (auto v = get_int(items, "triangle_id"))    { e.triangle_id    = *v; e.has_triangle_id    = true; }
    if (auto v = get_dbl(items, "simplex_weight")) { e.simplex_weight = *v; e.has_simplex_weight = true; }
    if (auto v = get_dbl(items, "g_00"))       { e.g_00       = *v; e.has_g_00       = true; }
    if (auto v = get_dbl(items, "g_01"))       { e.g_01       = *v; e.has_g_01       = true; }
    if (auto v = get_dbl(items, "g_10"))       { e.g_10       = *v; e.has_g_10       = true; }
    if (auto v = get_dbl(items, "g_11"))       { e.g_11       = *v; e.has_g_11       = true; }
    if (auto v = get_dbl(items, "jacobian_w")) { e.jacobian_w = *v; e.has_jacobian_w = true; }
    return e;
}

} // namespace

// ── Public API ────────────────────────────────────────────────────────────────

std::unique_ptr<GmlGraph> parse_gml(const std::string& text) {
    Lexer lex(text);

    Token t1 = lex.next();
    if (t1.kind != TokKind::IDENT || t1.str != "graph") return nullptr;
    Token t2 = lex.next();
    if (t2.kind != TokKind::LBRACKET) return nullptr;

    auto items = parse_body(lex);

    auto g = std::make_unique<GmlGraph>();
    if (auto v = get_int(items, "directed")) g->directed = (*v != 0);
    if (auto v = get_int(items, "id"))       g->id       = *v;
    if (auto v = get_str(items, "label"))    g->label    = *v;
    if (auto v = get_str(items, "comment"))  g->comment  = *v;

    for (const auto& rb : items) {
        if (!rb.is_block) continue;
        if (rb.key == "node") g->nodes.push_back(build_node(rb.children));
        else if (rb.key == "edge") g->edges.push_back(build_edge(rb.children));
    }
    return g;
}

std::unique_ptr<GmlGraph> parse_gml_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return nullptr;
    std::string text((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());
    return parse_gml(text);
}

} // namespace khanary::world
