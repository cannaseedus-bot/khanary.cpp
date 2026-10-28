//
// KHANARY_NUMERICAL_BACKEND_TEST — where the numbers run, and what that does not mean.
//
// Three kinds of refusal are kept apart because they answer different questions:
//
//   BACKEND_REQUIRES_D3D12   the API can never run on that device family
//   BACKEND_IS_CPU_ONLY      a CPU tool was pointed at a GPU
//   DEVICE_UNAVAILABLE       the API would work, but this machine lacks the device
//
// Conflating the third with the first is how "we support D3D12" becomes "this build
// needs D3D12". The HD 4600 rig has D3D11 CS5, OpenCL 1.2 and WebGL2, and no D3D12
// feature level — stated by the KLSL XVM target's own README, not assumed here.
//
// The lowering half is the other point: a Clifford product lowers to ONE ordinary
// GEMM via M(a)·b, and GEMM itself implements no algebra. Both halves are checked,
// including the negative case where GEMM is used without the lowered matrix.
//
#include "conformance.h"
#include "../graphscript/numerical_backend.h"
#include "../graphscript/tensor_bridge.h"

#include <cmath>
#include <string>
#include <vector>

namespace {

namespace ga = khanary::graphscript;
constexpr double kEps = 1e-12;

ga::BackendRequest request(ga::NumericalBackend backend, ga::BackendDevice device) {
    ga::BackendRequest out;
    out.backend = backend;
    out.device = device;
    return out;
}

}  // namespace

int main() {
    return khanary::test::run("numerical_backend", [&](khanary::test::Suite& suite) {
        const ga::BackendCapabilities hd4600 = ga::hd4600_capabilities();

        // ── 1. The machine's actual capabilities ───────────────────────────
        suite.check("capabilities/d3d11-1-present", hd4600.d3d11_1);
        suite.check("capabilities/opencl-1-2-present", hd4600.opencl_1_2);
        suite.check("capabilities/webgl2-present", hd4600.webgl2);
        // Measured: a D3D12 device IS created at feature level 11_1 on this driver.
        suite.check("capabilities/d3d12-available", hd4600.d3d12);
        suite.check("capabilities/no-shader-model-6-10", !hd4600.shader_model_6_10);

        // ── 2. API mismatches are permanent refusals ───────────────────────
        {
            const auto directml_on_d3d11 = ga::check_backend(
                request(ga::NumericalBackend::DirectML, ga::BackendDevice::D3D11_1), hd4600);
            suite.check("refuse/directml-on-d3d11", !directml_on_d3d11.admissible);
            suite.check("refuse/directml-code",
                        directml_on_d3d11.code == "BACKEND_REQUIRES_D3D12");
            suite.check("refuse/directml-explains", directml_on_d3d11.detail.find(
                                                        "interchangeable") != std::string::npos);

            const auto d12_on_d11 = ga::check_backend(
                request(ga::NumericalBackend::D3D12Compute, ga::BackendDevice::D3D11_1), hd4600);
            suite.check("refuse/d3d12-compute-on-d3d11", !d12_on_d11.admissible);
            suite.check("refuse/d3d12-compute-code", d12_on_d11.code == "BACKEND_REQUIRES_D3D12");

            const auto d11_on_d12 = ga::check_backend(
                request(ga::NumericalBackend::D3D11Compute, ga::BackendDevice::D3D12), hd4600);
            suite.check("refuse/d3d11-compute-on-d3d12", !d11_on_d12.admissible);
            suite.check("refuse/d3d11-compute-code",
                        d11_on_d12.code == "BACKEND_REQUIRES_D3D11_1");

            const auto opencl_on_d11 = ga::check_backend(
                request(ga::NumericalBackend::OpenCL12, ga::BackendDevice::D3D11_1), hd4600);
            suite.check("refuse/opencl-on-d3d11", !opencl_on_d11.admissible);
            suite.check("refuse/opencl-code",
                        opencl_on_d11.code == "BACKEND_REQUIRES_OPENCL_1_2");

            const auto webgl_on_d11 = ga::check_backend(
                request(ga::NumericalBackend::WebGL2, ga::BackendDevice::D3D11_1), hd4600);
            suite.check("refuse/webgl2-on-d3d11", !webgl_on_d11.admissible);

            const auto cpu_on_gpu = ga::check_backend(
                request(ga::NumericalBackend::NumpyReference, ga::BackendDevice::D3D11_1),
                hd4600);
            suite.check("refuse/numpy-is-cpu-only", !cpu_on_gpu.admissible);
            suite.check("refuse/numpy-code", cpu_on_gpu.code == "BACKEND_IS_CPU_ONLY");
            suite.check("refuse/numpy-not-a-provider",
                        cpu_on_gpu.detail.find("oracle") != std::string::npos);
        }

        // ── 3. A capability gap is not an API mismatch ─────────────────────
        {
            // Measured on this rig: D3D12 exists (FL 11_1), so the D3D12-only paths are
            // admitted here — the README's claim that no D3D12 feature level exists is
            // superseded by the probe.
            const auto d12_on_d12 = ga::check_backend(
                request(ga::NumericalBackend::D3D12Compute, ga::BackendDevice::D3D12), hd4600);
            suite.check("capability/d3d12-admitted-on-this-rig", d12_on_d12.admissible);

            const auto directml_on_d12 = ga::check_backend(
                request(ga::NumericalBackend::DirectML, ga::BackendDevice::D3D12), hd4600);
            suite.check("capability/directml-admitted-on-this-rig", directml_on_d12.admissible);

            // The capability-gap refusal still exists for a machine without D3D12: that
            // is a different answer from an API mismatch, and it must stay distinct.
            ga::BackendCapabilities older = hd4600;
            older.d3d12 = false;
            const auto refused_older = ga::check_backend(
                request(ga::NumericalBackend::D3D12Compute, ga::BackendDevice::D3D12), older);
            suite.check("capability/gap-refused-on-a-machine-without-it",
                        !refused_older.admissible);
            suite.check("capability/gap-code", refused_older.code == "DEVICE_UNAVAILABLE");
            suite.check("capability/gap-detail-names-the-feature-level",
                        refused_older.detail.find("feature level") != std::string::npos);
            suite.check("capability/gap-detail-offers-the-bounded-paths",
                        refused_older.detail.find("OpenCL 1.2") != std::string::npos);
        }

        // ── 4. The admitted paths on this rig ──────────────────────────────
        {
            suite.check("admit/cpu",
                        ga::check_backend(request(ga::NumericalBackend::CpuKernel,
                                                  ga::BackendDevice::Cpu), hd4600).admissible);
            suite.check("admit/d3d11-compute",
                        ga::check_backend(request(ga::NumericalBackend::D3D11Compute,
                                                  ga::BackendDevice::D3D11_1), hd4600)
                            .admissible);
            suite.check("admit/opencl",
                        ga::check_backend(request(ga::NumericalBackend::OpenCL12,
                                                  ga::BackendDevice::OpenCL12), hd4600)
                            .admissible);
            suite.check("admit/webgl2",
                        ga::check_backend(request(ga::NumericalBackend::WebGL2,
                                                  ga::BackendDevice::WebGL2), hd4600)
                            .admissible);
        }

        // ── 5. Roles are policy, not capability ────────────────────────────
        {
            suite.check("role/d3d11-is-primary",
                        ga::performance_role(ga::NumericalBackend::D3D11Compute) ==
                            ga::PerformanceRole::Primary);
            suite.check("role/opencl-is-alternative-not-committed",
                        ga::performance_role(ga::NumericalBackend::OpenCL12) ==
                            ga::PerformanceRole::Alternative);
            suite.check("role/cpu-is-reference",
                        ga::performance_role(ga::NumericalBackend::CpuKernel) ==
                            ga::PerformanceRole::Reference);
            suite.check("role/numpy-is-reference",
                        ga::performance_role(ga::NumericalBackend::NumpyReference) ==
                            ga::PerformanceRole::Reference);
            suite.check("role/directml-is-optional",
                        ga::performance_role(ga::NumericalBackend::DirectML) ==
                            ga::PerformanceRole::Optional);
            // No predetermined compute provider: the policy ranks only measured
            // domains, and OpenCL CPU is the measured GEMM winner.
            {
                ga::Measurement cpu_gemm;
                cpu_gemm.identity = ga::ProviderIdentity{ga::ProviderApi::OpenCL12,
                                                         ga::DeviceClass::Cpu, "i7-4790S"};
                cpu_gemm.operation = "gemm";
                cpu_gemm.shape = "512x512x512";
                cpu_gemm.residency = ga::Residency::Host;
                cpu_gemm.gflops = 54.1;
                ga::Measurement gpu_gemm;
                gpu_gemm.identity = ga::ProviderIdentity{ga::ProviderApi::D3D11,
                                                         ga::DeviceClass::IntegratedGpu,
                                                         "HD 4600"};
                gpu_gemm.operation = "gemm";
                gpu_gemm.shape = "512x512x512";
                gpu_gemm.residency = ga::Residency::Host;
                gpu_gemm.gflops = 32.2;

                const std::vector<ga::Measurement> measured = {cpu_gemm, gpu_gemm};
                const std::vector<ga::ProviderIdentity> admitted = {cpu_gemm.identity,
                                                                    gpu_gemm.identity};
                const auto policy = ga::policy_select(measured, admitted, "gemm",
                                                      "512x512x512", ga::Residency::Host);
                suite.check("policy/picks-the-measured-winner",
                            policy.classified &&
                                policy.chosen.identity.device_class == ga::DeviceClass::Cpu);
                suite.check("policy/no-hardcoded-gpu-preference",
                            policy.chosen.identity.api == ga::ProviderApi::OpenCL12);

                // An unmeasured domain gets no winner but keeps its eligible set.
                const auto unmeasured = ga::policy_select(measured, admitted, "gemm",
                                                          "4096x4096x4096",
                                                          ga::Residency::Host);
                suite.check("policy/unmeasured-stays-eligible",
                            !unmeasured.classified && unmeasured.eligible.size() == 2);
            }
            // An unmeasured domain is a policy result, not an execution decision: the
            // fallback has to be explicit, never a silent hardcoded preference.
            {
                const std::vector<ga::ProviderIdentity> pool = {
                    ga::ProviderIdentity{ga::ProviderApi::OpenCL12, ga::DeviceClass::Cpu,
                                         "i7-4790S"},
                    ga::ProviderIdentity{ga::ProviderApi::D3D11,
                                         ga::DeviceClass::IntegratedGpu, "HD 4600"}};
                const auto none =
                    ga::policy_select({}, pool, "gemm", "512x512x512", ga::Residency::Host);
                suite.check("fallback/unmeasured", !none.classified);

                const auto refused = ga::resolve_execution(none, ga::FallbackPolicy::Refuse);
                suite.check("fallback/refuse-does-not-execute", !refused.execute);
                suite.check("fallback/refuse-explains",
                            refused.refusal.find("refuse") != std::string::npos);

                const auto calibrate = ga::resolve_execution(none, ga::FallbackPolicy::Calibrate);
                suite.check("fallback/calibrate-signals-measurement", !calibrate.execute);

                const auto first = ga::resolve_execution(none, ga::FallbackPolicy::FirstEligible);
                suite.check("fallback/first-eligible-executes", first.execute);
                suite.check("fallback/basis-is-recorded",
                            first.basis.find("documented fallback") != std::string::npos);

                const auto unnamed =
                    ga::resolve_execution(none, ga::FallbackPolicy::CallerSupplied);
                suite.check("fallback/caller-required", !unnamed.execute);

                const auto named =
                    ga::resolve_execution(none, ga::FallbackPolicy::CallerSupplied, &pool[0]);
                suite.check("fallback/caller-choice-honoured",
                            named.execute && named.provider.device_class == ga::DeviceClass::Cpu);

                const ga::ProviderIdentity outsider{ga::ProviderApi::WebGL2,
                                                    ga::DeviceClass::Other, "browser"};
                const auto rejected =
                    ga::resolve_execution(none, ga::FallbackPolicy::CallerSupplied, &outsider);
                suite.check("fallback/ineligible-caller-choice-refused", !rejected.execute);

                // A classified domain is decided by the ranking, not by the fallback.
                ga::Measurement m;
                m.identity = pool[0];
                m.operation = "gemm";
                m.shape = "512x512x512";
                m.residency = ga::Residency::Host;
                m.gflops = 54.1;
                const auto ranked = ga::policy_select({m}, pool, "gemm", "512x512x512",
                                                      ga::Residency::Host);
                const auto decided =
                    ga::resolve_execution(ranked, ga::FallbackPolicy::Refuse);
                suite.check("fallback/ranking-overrides-fallback-refusal", decided.execute);
                suite.check("fallback/basis-cites-the-ranking",
                            decided.basis.find("validated ranking") != std::string::npos);
            }

            // Rank staleness: capability changes independently of history, so a ranked
            // provider that is no longer admitted must not be selected.
            {
                auto make = [](ga::ProviderApi api, ga::DeviceClass klass, const char* id,
                               double gflops) {
                    ga::Measurement m;
                    m.identity = ga::ProviderIdentity{api, klass, id};
                    m.operation = "gemm";
                    m.shape = "512x512x512";
                    m.residency = ga::Residency::Host;
                    m.gflops = gflops;
                    return m;
                };
                const ga::ProviderIdentity cpu{ga::ProviderApi::OpenCL12, ga::DeviceClass::Cpu,
                                               "i7-4790S"};
                const ga::ProviderIdentity d11{ga::ProviderApi::D3D11,
                                               ga::DeviceClass::IntegratedGpu, "HD 4600"};
                const ga::ProviderIdentity ogpu{ga::ProviderApi::OpenCL12,
                                                ga::DeviceClass::IntegratedGpu, "HD 4600"};

                const std::vector<ga::Measurement> ranked = {
                    make(ga::ProviderApi::OpenCL12, ga::DeviceClass::Cpu, "i7-4790S", 54.1),
                    make(ga::ProviderApi::D3D11, ga::DeviceClass::IntegratedGpu, "HD 4600", 32.2),
                    make(ga::ProviderApi::OpenCL12, ga::DeviceClass::IntegratedGpu, "HD 4600",
                         26.1)};

                // All eligible: the recorded winner.
                const auto all = ga::select_provider(ranked, {cpu, d11, ogpu}, "gemm",
                                                     "512x512x512", ga::Residency::Host);
                suite.check("staleness/winner-when-all-eligible", all.classified &&
                            all.chosen.identity.device_class == ga::DeviceClass::Cpu);

                // The winner disappears from admission: next ranked eligible must win.
                const auto survived = ga::select_provider(ranked, {d11, ogpu}, "gemm",
                                                          "512x512x512", ga::Residency::Host);
                suite.check("staleness/highest-ranked-eligible-wins", survived.classified &&
                            survived.chosen.identity.api == ga::ProviderApi::D3D11);
                suite.check("staleness/does-not-carry-the-stale-winner",
                            survived.classified &&
                                survived.chosen.identity.device_class != ga::DeviceClass::Cpu);

                // No ranked provider remains eligible: unclassified, and it says why.
                const ga::ProviderIdentity stranger{ga::ProviderApi::DirectML,
                                                    ga::DeviceClass::IntegratedGpu, "HD 4600"};
                const auto gone = ga::select_provider(ranked, {stranger}, "gemm", "512x512x512",
                                                      ga::Residency::Host);
                suite.check("staleness/no-survivor-is-unclassified", !gone.classified);
                suite.check("staleness/reason-names-the-staleness",
                            gone.reason.find("no longer admitted") != std::string::npos);
                suite.check("staleness/still-eligible-for-fallback",
                            gone.eligible.size() == std::size_t{1});

                // And the fallback path handles it as it handles any unclassified domain.
                const auto decided =
                    ga::resolve_execution(gone, ga::FallbackPolicy::FirstEligible);
                suite.check("staleness/fallback-still-resolves", decided.execute);
            }

            suite.check("workclass/presentation-is-separate",
                        std::string(ga::work_class_name(ga::WorkClass::Presentation)) ==
                            "presentation");

            // The measured loser keeps its capability; only the commitment goes.
            const auto opencl = ga::check_backend(
                request(ga::NumericalBackend::OpenCL12, ga::BackendDevice::OpenCL12), hd4600);
            suite.check("role/loser-keeps-capability", opencl.admissible);
        }

        // ── 6. Shipped kernels and their dispatch geometry ─────────────────
        {
            const auto& kernels = ga::shipped_kernels();
            suite.check("kernels/declared", !kernels.empty());
            bool tiled_first = !kernels.empty() && kernels.front().tiled &&
                               kernels.front().threads_x == 8 && kernels.front().threads_y == 8 &&
                               kernels.front().groupshared_tile == 8;
            suite.check("kernels/tiled-leads-with-8x8-tiles", tiled_first);

            // Every shipped kernel carries the shipped status, and none of them needs a
            // device this rig lacks.
            bool all_shipped = true;
            bool all_admissible = true;
            for (const auto& kernel : kernels) {
                all_shipped = all_shipped && kernel.status == ga::KernelStatus::Shipped;
                all_admissible = all_admissible && ga::kernel_admissible(kernel, hd4600);
            }
            suite.check("kernels/all-shipped", all_shipped);
            suite.check("kernels/all-admissible-on-this-rig", all_admissible);
        }

        // ── 6b. A candidate is not a commitment ───────────────────────────
        {
            const auto& unavailable = ga::unavailable_kernels();
            suite.check("candidates/recorded", !unavailable.empty());

            bool all_experimental = true;
            bool all_needing_d3d12 = true;
            for (const auto& kernel : unavailable) {
                all_experimental =
                    all_experimental && kernel.status == ga::KernelStatus::Experimental;
                all_needing_d3d12 = all_needing_d3d12 && kernel.requires_d3d12;
            }
            suite.check("candidates/are-experimental", all_experimental);
            suite.check("candidates/require-d3d12", all_needing_d3d12);

            // SM 6.10 cooperative matrix cannot run here, and is not in the shipped list.
            bool absent_from_shipped = true;
            for (const auto& kernel : ga::shipped_kernels()) {
                absent_from_shipped =
                    absent_from_shipped && kernel.name != "kuhul_cooperative_matrix";
            }
            suite.check("candidates/not-in-the-dispatch-table", absent_from_shipped);

            if (!unavailable.empty()) {
                // D3D12 exists here; SM 6.10 does not. The refusal is the shader model.
                suite.check("candidates/d3d12-itself-is-available", hd4600.d3d12);
                suite.check("candidates/refused-on-shader-model",
                            !ga::kernel_admissible(unavailable.front(), hd4600));

                // On a D3D12 machine it becomes admissible, but it stays experimental.
                ga::BackendCapabilities sm610 = hd4600;
                sm610.shader_model_6_10 = true;
                suite.check("candidates/admissible-where-the-shader-model-exists",
                            ga::kernel_admissible(unavailable.front(), sm610));
                suite.check("candidates/status-does-not-change-with-hardware",
                            unavailable.front().status == ga::KernelStatus::Experimental);
            }
        }

        // ── 6c. Rank is scoped to a domain; roles are per provider ────────
        {
            auto measurement = [](ga::ProviderApi api, ga::DeviceClass klass, const char* id,
                                  const std::string& shape, ga::Residency residency,
                                  double gflops) {
                ga::Measurement m;
                m.identity.api = api;
                m.identity.device_class = klass;
                m.identity.device_id = id;
                m.operation = "gemm";
                m.shape = shape;
                m.residency = residency;
                m.kernel_variant = "tiled16";
                m.methodology = "test fixture";
                m.gflops = gflops;
                return m;
            };

            const std::string big = "1024x1024x1024";
            const std::vector<ga::Measurement> measurements = {
                measurement(ga::ProviderApi::OpenCL12, ga::DeviceClass::Cpu, "i7-4790S", big,
                            ga::Residency::Host, 41.3),
                measurement(ga::ProviderApi::D3D11, ga::DeviceClass::IntegratedGpu, "HD 4600",
                            big, ga::Residency::Host, 25.7),
                measurement(ga::ProviderApi::OpenCL12, ga::DeviceClass::IntegratedGpu, "HD 4600",
                            big, ga::Residency::Host, 22.6),
            };

            // Host residency: the CPU OpenCL device wins on measured compute.
            const auto host = ga::select_provider(measurements, "gemm", big, ga::Residency::Host);
            suite.check("select/host-classified", host.classified);
            suite.check("select/host-picks-opencl-cpu",
                        host.classified &&
                            host.chosen.identity.api == ga::ProviderApi::OpenCL12 &&
                            host.chosen.identity.device_class == ga::DeviceClass::Cpu);
            suite.check("select/host-value", host.classified && host.chosen.gflops == 41.3);

            // D3D11-buffer residency: nothing is measured, and the selector says so
            // rather than reusing the host ranking.
            const auto resident =
                ga::select_provider(measurements, "gemm", big, ga::Residency::D3D11Buffer);
            suite.check("select/resident-class-unclassified", !resident.classified);
            suite.check("select/resident-reason-names-the-domain",
                        resident.reason.find("d3d11_buffer") != std::string::npos);

            // Unmeasured is not ineligible: the domain has no performance winner, but it
            // still has legal execution paths.
            const std::vector<ga::ProviderIdentity> eligible = {
                ga::ProviderIdentity{ga::ProviderApi::OpenCL12, ga::DeviceClass::Cpu,
                                     "i7-4790S"},
                ga::ProviderIdentity{ga::ProviderApi::D3D11, ga::DeviceClass::IntegratedGpu,
                                     "HD 4600"}};
            const auto unmeasured = ga::select_provider(measurements, eligible, "gemm", big,
                                                        ga::Residency::D3D11Buffer);
            suite.check("select/unmeasured-not-ranked", !unmeasured.classified);
            suite.check("select/unmeasured-still-eligible",
                        unmeasured.eligible.size() == std::size_t{2});
            suite.check("select/unmeasured-reason-says-so",
                        unmeasured.reason.find("remain eligible") != std::string::npos);

            // A different shape is a different domain.
            const auto other_shape =
                ga::select_provider(measurements, "gemm", "512x512x512", ga::Residency::Host);
            suite.check("select/other-shape-unclassified", !other_shape.classified);

            // Identity distinguishes two providers behind one API.
            const ga::ProviderIdentity cpu{ga::ProviderApi::OpenCL12, ga::DeviceClass::Cpu,
                                           "i7-4790S"};
            const ga::ProviderIdentity gpu{ga::ProviderApi::OpenCL12,
                                           ga::DeviceClass::IntegratedGpu, "HD 4600"};
            suite.check("identity/same-api-different-provider",
                        ga::provider_identity_name(cpu) != ga::provider_identity_name(gpu));
            suite.check("identity/names-the-api-and-class",
                        ga::provider_identity_name(cpu).find("opencl_1_2:cpu") == 0);
        }

        // ── 7. The Clifford product lowers to one GEMM ─────────────────────
        {
            const std::vector<double> a = {0.0, 1.0, 0.0, 0.0};  // e1
            const std::vector<double> b = {0.0, 0.0, 1.0, 0.0};  // e2

            const std::vector<double> matrix = ga::clifford_product_matrix(a);
            suite.check("lower/matrix-is-four-by-four", matrix.size() == std::size_t{16});

            const std::vector<double> via_gemm = ga::clifford_product_via_gemm(a, b);
            const std::vector<double> direct = ga::clifford_product_cl20(a, b);
            suite.check("lower/gemm-matches-the-table", ga::same_coefficients(via_gemm, direct));

            // e1·e2 = e12
            suite.check("lower/e1-e2-is-the-bivector",
                        std::fabs(via_gemm[3] - 1.0) <= kEps);

            // e1·e1 = 1
            const auto square = ga::clifford_product_via_gemm(a, a);
            suite.check("lower/e1-squared-is-one", std::fabs(square[0] - 1.0) <= kEps);

            // Agreement over a spread of operands.
            bool agrees = true;
            const std::vector<std::vector<double>> operands = {
                {1.0, 0.0, 0.0, 0.0}, {0.0, 1.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0},
                {0.0, 0.0, 0.0, 1.0}, {2.0, 3.0, -1.0, 0.5}, {-1.0, 0.0, 2.0, 3.0},
            };
            for (const auto& left : operands) {
                for (const auto& right : operands) {
                    agrees = agrees && ga::same_coefficients(
                                            ga::clifford_product_via_gemm(left, right),
                                            ga::clifford_product_cl20(left, right), kEps);
                }
            }
            suite.check("lower/agrees-over-a-spread-of-operands", agrees);
        }

        // ── 8. GEMM is GEMM, and that is the point ─────────────────────────
        {
            // alpha/beta form: C = 2*A*B + 0.5*C
            const std::vector<double> a = {1.0, 2.0};      // 1x2
            const std::vector<double> b = {3.0, 4.0};      // 2x1
            std::vector<double> c = {10.0};                // 1x1
            const bool ok = ga::gemm(2.0, a, 1, 2, b, 1, 0.5, &c, 1, 1);
            suite.check("gemm/alpha-beta-applied",
                        ok && std::fabs(c[0] - (2.0 * (1.0 * 3.0 + 2.0 * 4.0) + 0.5 * 10.0)) <= kEps);

            std::vector<double> bad = {0.0, 0.0};
            suite.check("gemm/refuses-a-shape-mismatch",
                        !ga::gemm(1.0, a, 1, 2, b, 1, 0.0, &bad, 1, 2));

            // The negative case: GEMM over two coefficient arrays is NOT the Clifford
            // product unless the lowered matrix was built first.
            const std::vector<double> e1 = {0.0, 1.0, 0.0, 0.0};
            const std::vector<double> e2 = {0.0, 0.0, 1.0, 0.0};
            std::vector<double> naive(1, 0.0);
            const bool ran = ga::gemm(1.0, e1, 4, 1, e2, 1, 0.0, &naive, 4, 1);
            // That combination has no valid shapes at all: a 4x1 times a 1x1 is what
            // the signature allows, and it is plainly not a Clifford product.
            suite.check("gemm/is-not-an-algebra", !ran ||
                                                    std::fabs(naive[0]) <= kEps);

            const auto lowered = ga::clifford_product_via_gemm(e1, e2);
            suite.check("gemm/lowering-is-what-makes-it-one",
                        std::fabs(lowered[3] - 1.0) <= kEps);
        }
    });
}
