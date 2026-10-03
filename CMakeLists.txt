cmake_minimum_required(VERSION 3.28)
project(KHANARY CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT MSVC)
    message(FATAL_ERROR "KHANARY requires MSVC 2022+")
endif()

add_compile_options(/W4 /WX /utf-8 /permissive- /Zc:__cplusplus)
add_compile_definitions(UNICODE _UNICODE NOMINMAX WIN32_LEAN_AND_MEAN)
add_compile_definitions(KHANARY_HD4600_CEILING=1879048192)

add_executable(KHANARY
    src/main.cpp
    src/manifest/manifest_registry.cpp
    src/manifest/manifest_loader.cpp
    src/manifest/scx_resolver.cpp
    src/kuhul/phase_engine.cpp
    src/kuhul/fold_engine.cpp
    src/kuhul/fold_direction.cpp
    src/kuhul/kuhul_vm.cpp
    src/kuhul/phase_driver_lock.cpp
    src/kuhul/xcfe_executor.cpp
    src/kuhul/gram_store.cpp
    src/kuhul/ngram_phase_classes.cpp
    src/klsl/klsl_lexer.cpp
    src/klsl/klsl_parser.cpp
    src/klsl/klsl_ir.cpp
    src/klsl/hlsl_codegen.cpp
    src/klsl/wgsl_codegen.cpp
    src/klsl/d3dcompile_bridge.cpp
    src/gpu/d3d11_device.cpp
    src/gpu/dml_ops.cpp
    src/gpu/cso_integrity.cpp
    src/gpu/unified_tensor.cpp
    src/gpu/tensor_buffer.cpp
    src/gpu/compute_dispatch.cpp
    src/gpu/gram_gpu_scorer.cpp
    src/gpu/svg_compute_runtime.cpp
    src/gpu/dds_loader.cpp
    src/gpu/sdds_format.cpp
    src/gpu/hot_swap.cpp
    src/inference/tokenizer.cpp
    src/inference/moe_router.cpp
    src/inference/inference_engine.cpp
    src/inference/semantic_attention.cpp
    src/inference/kv_cache.cpp
    src/inference/sampler.cpp
    src/inference/fold_map.cpp
    src/decision/decision_mu.cpp
    src/micronaut/micronaut_registry.cpp
    src/micronaut/aiml_engine.cpp
    src/micronaut/eliza_operators.cpp
    src/micronaut/relation_isa.cpp
    src/micronaut/track_loader.cpp
    src/micronaut/mx2lm-ngram.cpp
    src/micronaut/sandbox_inspector.cpp
    src/micronaut/gravity_field.cpp
    src/micronaut/greeting_style.cpp
    src/micronaut/control_flow.cpp
    src/micronaut/mou_router.cpp
    src/micronaut/micronaut_transpiler.cpp
    src/world/world_state.cpp
    src/world/field_dag.cpp
    src/world/gml_parser.cpp
    src/world/gyroscope.cpp
    src/world/vector_projection.cpp
    src/micronaut/factory.cpp
    src/memory/perception.cpp
    src/memory/confidence.cpp
    src/memory/semantic_graph.cpp
    src/memory/scoring.cpp
    src/io/json_rpc_server.cpp
    src/io/mcp_server.cpp
    src/io/code_parser.cpp
    src/io/jrom.cpp
    src/io/kql.cpp
    src/io/projection.cpp
    src/io/runtime_attention.cpp
    src/io/request_router.cpp
    src/io/svg_tensor.cpp
    src/io/stb_map_convert.cpp
    src/gpu/stb_map.cpp
    src/util/sha256.cpp
    src/util/json.cpp
    src/util/schema.cpp
    src/util/canonical_hash.cpp
    src/util/thread_pool.cpp
)

target_include_directories(KHANARY PRIVATE src)

target_link_libraries(KHANARY PRIVATE
    d3d11
    dxgi
    d3dcompiler
    dxguid
    Ws2_32
    Shlwapi
)

# DirectML — locate via vcpkg or manual path
find_library(DML_LIB DirectML HINTS
    "$ENV{DIRECTML_PATH}/bin/x64-win"
    "C:/Program Files/Microsoft DirectML/bin/x64-win"
    "${CMAKE_CURRENT_SOURCE_DIR}/extern/DirectML/bin/x64-win"
)
find_path(DML_INCLUDE_DIR DirectML.h HINTS
    "$ENV{DIRECTML_PATH}/include"
    "C:/Program Files/Microsoft DirectML/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/extern/DirectML/include"
)
if(DML_LIB AND DML_INCLUDE_DIR)
    target_include_directories(KHANARY PRIVATE ${DML_INCLUDE_DIR})
    target_link_libraries(KHANARY PRIVATE ${DML_LIB})
    target_compile_definitions(KHANARY PRIVATE KHANARY_HAS_DIRECTML)
else()
    message(WARNING "DirectML not found — DML_OPS will be disabled at compile time")
endif()

install(TARGETS KHANARY DESTINATION bin)
install(DIRECTORY data/ DESTINATION data)

# Projection acceptance tests — lightweight, no GPU/D3D deps
add_executable(KHANARY_PTEST
    src/tests/test_projection.cpp
    src/io/projection.cpp
    src/io/jrom.cpp
    src/world/gyroscope.cpp
    src/util/sha256.cpp
    src/util/json.cpp
)
target_include_directories(KHANARY_PTEST PRIVATE src)

add_executable(KHANARY_JROM_TEST
    src/tests/test_jrom.cpp
    src/io/jrom.cpp
    src/util/json.cpp
    src/util/sha256.cpp
)
target_include_directories(KHANARY_JROM_TEST PRIVATE src)

add_executable(KHANARY_KQL_TEST
    src/tests/test_kql.cpp
    src/io/kql.cpp
    src/util/json.cpp
    src/util/sha256.cpp
)
target_include_directories(KHANARY_KQL_TEST PRIVATE src)

add_executable(KHANARY_AIML_TEST
    src/tests/test_aiml_graph.cpp
    src/micronaut/aiml_engine.cpp
    src/micronaut/eliza_operators.cpp
    src/micronaut/relation_isa.cpp
    src/util/json.cpp
)
target_include_directories(KHANARY_AIML_TEST PRIVATE src)

add_executable(KHANARY_ATTENTION_TEST
    src/tests/test_runtime_attention.cpp
    src/io/runtime_attention.cpp
)
target_include_directories(KHANARY_ATTENTION_TEST PRIVATE src)

add_executable(KHANARY_ROUTER_TEST
    src/tests/test_router_policy.cpp
    src/io/request_router.cpp
    src/io/runtime_attention.cpp
    src/decision/decision_mu.cpp
    src/inference/moe_router.cpp
    src/micronaut/factory.cpp
    src/micronaut/micronaut_registry.cpp
    src/micronaut/mou_router.cpp
    src/micronaut/micronaut_transpiler.cpp
    src/micronaut/relation_isa.cpp
    src/micronaut/gravity_field.cpp
    src/memory/perception.cpp
    src/memory/confidence.cpp
    src/memory/semantic_graph.cpp
    src/memory/scoring.cpp
    src/kuhul/phase_engine.cpp
    src/util/json.cpp
)
target_include_directories(KHANARY_ROUTER_TEST PRIVATE src)

add_executable(KHANARY_NGRAM_TEST
    src/tests/test_mx2lm_gram_channels.cpp
    src/micronaut/mx2lm-ngram.cpp
    src/micronaut/control_flow.cpp
    src/micronaut/eliza_operators.cpp
    src/micronaut/relation_isa.cpp
    src/kuhul/phase_engine.cpp
    src/kuhul/gram_store.cpp
    src/kuhul/ngram_phase_classes.cpp
    src/util/json.cpp
    src/util/sha256.cpp
)
target_include_directories(KHANARY_NGRAM_TEST PRIVATE src)

enable_testing()
add_test(
    NAME KHANARY_PTEST
    COMMAND $<TARGET_FILE:KHANARY_PTEST>
            "${CMAKE_SOURCE_DIR}/data/projection_abi_v1.json"
            "${CMAKE_SOURCE_DIR}/data/gyro_abi_v1.json")
add_test(
    NAME KHANARY_JROM_TEST
    COMMAND $<TARGET_FILE:KHANARY_JROM_TEST>)
add_test(
    NAME KHANARY_KQL_TEST
    COMMAND $<TARGET_FILE:KHANARY_KQL_TEST>)
add_test(
    NAME KHANARY_AIML_TEST
    COMMAND $<TARGET_FILE:KHANARY_AIML_TEST>)
add_test(
    NAME KHANARY_ATTENTION_TEST
    COMMAND $<TARGET_FILE:KHANARY_ATTENTION_TEST>)
add_test(
    NAME KHANARY_ROUTER_TEST
    COMMAND $<TARGET_FILE:KHANARY_ROUTER_TEST>)
add_test(
    NAME KHANARY_NGRAM_TEST
    COMMAND $<TARGET_FILE:KHANARY_NGRAM_TEST>)

# Shared deterministic verdict ABI for every native suite.
target_sources(KHANARY_AIML_TEST PRIVATE src/util/sha256.cpp)
target_sources(KHANARY_ATTENTION_TEST PRIVATE src/util/json.cpp src/util/sha256.cpp)
target_sources(KHANARY_ROUTER_TEST PRIVATE src/util/sha256.cpp)

add_executable(KHANARY_CONTRACT_TEST
    src/tests/test_contracts.cpp src/util/json.cpp src/util/schema.cpp src/util/sha256.cpp
    src/manifest/manifest_registry.cpp src/micronaut/track_loader.cpp
    src/micronaut/micronaut_registry.cpp src/micronaut/relation_isa.cpp
    src/micronaut/gravity_field.cpp
    src/kuhul/phase_engine.cpp src/kuhul/xcfe_executor.cpp)
target_include_directories(KHANARY_CONTRACT_TEST PRIVATE src)
add_test(NAME KHANARY_CONTRACT_TEST COMMAND KHANARY_CONTRACT_TEST
    "${CMAKE_SOURCE_DIR}/data/conformance/contracts.json" "${CMAKE_SOURCE_DIR}/data")

add_executable(KHANARY_CANONICAL_TEST src/tests/test_canonical.cpp
    src/util/json.cpp src/util/sha256.cpp src/util/canonical_hash.cpp)
target_include_directories(KHANARY_CANONICAL_TEST PRIVATE src)
add_test(NAME KHANARY_CANONICAL_TEST COMMAND KHANARY_CANONICAL_TEST "${CMAKE_SOURCE_DIR}/data/conformance/canonical.json")

add_executable(KHANARY_SDDS_TEST src/tests/test_sdds.cpp src/gpu/sdds_format.cpp
    src/util/json.cpp src/util/sha256.cpp)
target_include_directories(KHANARY_SDDS_TEST PRIVATE src)
add_test(NAME KHANARY_SDDS_TEST COMMAND KHANARY_SDDS_TEST)

add_executable(KHANARY_GPU_DEVICE_TEST src/tests/test_gpu_device.cpp
    src/gpu/d3d11_device.cpp src/util/json.cpp src/util/sha256.cpp)
target_include_directories(KHANARY_GPU_DEVICE_TEST PRIVATE src)
target_link_libraries(KHANARY_GPU_DEVICE_TEST PRIVATE d3d11 dxgi dxguid)
add_test(NAME KHANARY_GPU_DEVICE_TEST COMMAND KHANARY_GPU_DEVICE_TEST)

add_executable(KHANARY_TENSOR_TEST src/tests/test_unified_tensor.cpp
    src/gpu/unified_tensor.cpp src/gpu/tensor_buffer.cpp src/gpu/d3d11_device.cpp
    src/gpu/dml_ops.cpp src/util/json.cpp src/util/sha256.cpp)
target_include_directories(KHANARY_TENSOR_TEST PRIVATE src)
target_link_libraries(KHANARY_TENSOR_TEST PRIVATE d3d11 dxgi dxguid d3dcompiler)
add_test(NAME KHANARY_TENSOR_TEST COMMAND KHANARY_TENSOR_TEST)

add_executable(KHANARY_CSO_INTEGRITY_TEST src/tests/test_cso_integrity.cpp
    src/gpu/cso_integrity.cpp src/util/json.cpp src/util/sha256.cpp)
target_include_directories(KHANARY_CSO_INTEGRITY_TEST PRIVATE src)
add_test(NAME KHANARY_CSO_INTEGRITY_TEST COMMAND KHANARY_CSO_INTEGRITY_TEST)

add_executable(KHANARY_GML_PARSER_TEST src/tests/test_gml_parser.cpp
    src/world/gml_parser.cpp src/util/json.cpp src/util/sha256.cpp)
target_include_directories(KHANARY_GML_PARSER_TEST PRIVATE src)
add_test(NAME KHANARY_GML_PARSER_TEST COMMAND KHANARY_GML_PARSER_TEST)

add_executable(KHANARY_HYPERBOLIC_ATTENTION_TEST src/tests/test_hyperbolic_attention.cpp
    src/world/vector_projection.cpp src/world/gml_parser.cpp
    src/gpu/gram_gpu_scorer.cpp src/gpu/tensor_buffer.cpp src/gpu/d3d11_device.cpp
    src/gpu/stb_map.cpp src/gpu/sdds_format.cpp src/kuhul/gram_store.cpp
    src/util/json.cpp src/util/sha256.cpp)
target_include_directories(KHANARY_HYPERBOLIC_ATTENTION_TEST PRIVATE src)
target_link_libraries(KHANARY_HYPERBOLIC_ATTENTION_TEST PRIVATE d3d11 dxgi dxguid d3dcompiler)
add_test(NAME KHANARY_HYPERBOLIC_ATTENTION_TEST COMMAND KHANARY_HYPERBOLIC_ATTENTION_TEST)

add_executable(KHANARY_SEMANTIC_ATTENTION_TEST src/tests/test_semantic_attention.cpp
    src/inference/semantic_attention.cpp
    src/gpu/gram_gpu_scorer.cpp src/gpu/tensor_buffer.cpp src/gpu/d3d11_device.cpp
    src/gpu/stb_map.cpp src/gpu/sdds_format.cpp src/kuhul/gram_store.cpp
    src/util/json.cpp src/util/sha256.cpp)
target_include_directories(KHANARY_SEMANTIC_ATTENTION_TEST PRIVATE src)
target_link_libraries(KHANARY_SEMANTIC_ATTENTION_TEST PRIVATE d3d11 dxgi dxguid d3dcompiler)
add_test(NAME KHANARY_SEMANTIC_ATTENTION_TEST COMMAND KHANARY_SEMANTIC_ATTENTION_TEST)

# MetricTensor (0,2) tensor: euclidean/diagonal factories, arc_weight equivalence,
# Sylvester positive-definite check.  Header-only; depends only on phase_engine.
add_executable(KHANARY_METRIC_TENSOR_TEST src/tests/test_metric_tensor.cpp
    src/kuhul/phase_engine.cpp src/util/json.cpp src/util/sha256.cpp)
target_include_directories(KHANARY_METRIC_TENSOR_TEST PRIVATE src)
add_test(NAME KHANARY_METRIC_TENSOR_TEST COMMAND KHANARY_METRIC_TENSOR_TEST)

find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    add_test(NAME KHANARY_SCHEMA_TEST COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/check_contracts.py")
    add_test(NAME KHANARY_GEOMETRY_ASSET_TEST COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/check_geometry_model.py")
    # Control-flow grammar: the EBNF control contract must agree with the
    # runtime-emitted control_flow_row artifact (and the Python mirror).
    add_test(NAME KHANARY_CONTROLFLOW_GRAMMAR_TEST
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/grammar/mini_transpilers.py" control-check)
    # Fold/phase invariants promised by docs/GLOSSARY.md must match the
    # implementation (phase geometry, engine arity, formal-model strings).
    add_test(NAME KHANARY_FOLD_GEOMETRY_TEST
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/grammar/mini_transpilers.py" fold-geometry-check)
    # K'UHUL v8 standalone/runtime grammar: tensor bundles, hyperbolic events,
    # CSO/CDATA dispatch, Einstein contraction, and semantic-vs-backend fold namespaces.
    add_test(NAME KHANARY_KUHUL_GRAMMAR_TEST
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/check_kuhul_grammar.py")
    # Fold Binding layer: K'UHUL owns alpha/theta; backend entries must satisfy
    # per-phase Hoare contracts before they can execute as projections.
    add_test(NAME KHANARY_FOLD_BINDING_TEST
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/check_fold_bindings.py")
    # XCFE Policy grammar: policy is declarative data; XCFE evaluates
    # Pi(Evidence, State, Event) into the next state and route.
    add_test(NAME KHANARY_XCFE_POLICY_TEST
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/check_xcfe_policy.py")
    # Fold Kernel lifecycle: state-dependent JSON kernel records grow from
    # learned n-grams through binding/evidence/policy into XCFE-owned routes.
    add_test(NAME KHANARY_FOLD_KERNEL_TEST
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/check_fold_kernel.py")
    # The capability-gap artifact must stay current with the operator table and
    # the tracks' declared capabilities.
    add_test(NAME KHANARY_CAPABILITY_GAP_TEST
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/grammar/mini_transpilers.py" capability-gap --check)
    # Code-grammar artifacts: <Source, Freshness, Shape> per artifact, with a
    # missing canonical source reported as UNVERIFIED rather than invented.
    add_test(NAME KHANARY_CODE_GRAMMAR_TEST
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/grammar/mini_transpilers.py" code-grammar-check)
    # GML semantic validator: ParsedGraph → VerifiedGraph boundary.
    # Validates all checks in data/grammar/gml.validator.json against
    # data/conformance/gml_validator.json fixtures and shipped data/*.gml files.
    add_test(NAME KHANARY_GML_VALIDATOR_TEST
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/check_gml_validator.py")
    # Episode journal and provenance contract validator.
    # Checks episode.contract.json structure and scoring/verdict algebraic invariants.
    add_test(NAME KHANARY_EPISODE_TEST
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/check_episode.py")
    # Field-vector metadata validator for data/world/field.xml.
    # Coords/source are hard-checked; tensor and QKV are partial (warn-only).
    add_test(NAME KHANARY_FIELD_VECTOR_TEST
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/check_field_vectors.py")
    # VECTOR-µ provider registry: GML topology before derived mesh/vector geometry.
    add_test(NAME KHANARY_VECTORS_XML_TEST
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/check_vectors_xml.py")
    # Dataset ingestion lane: external corpora become candidate gram/key/vector
    # evidence; dictionary key-grammar remains the authority reference.
    add_test(NAME KHANARY_DATASET_INGESTION_TEST
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/check_dataset_ingestion.py")
    # METRIC-µ: per-track local metric fields plus explicit mesh bridges;
    # VECTOR-µ remains a provider, not semantic authority.
    add_test(NAME KHANARY_METRIC_GRAPH_TEST
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/check_metric_graph.py")
    # STB/Tensor Micronaut projection contracts. STB brains are geometry/evidence
    # containers; tensor slots give every known Micronaut a fiber address without
    # promoting unsupported private brains.
    add_test(NAME KHANARY_STB_TENSOR_BRAINS_TEST
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/check_stb_tensor_brains.py")
    add_test(NAME KHANARY_MICRONAUT_TENSOR_SLOTS_TEST
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/check_micronaut_tensor_slots.py")
    add_test(NAME KHANARY_MICRONAUT_GRAMMAR_TEST
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/check_micronaut_grammar.py")
    add_test(NAME KHANARY_ATTENTION_SHADER_CONTRACT_TEST
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/check_attention_shader.py")
    # CSS key-grammar: every ⟁KEY in css.manifest.json must have a compiled
    # JSONL entry, with expands_to references and inline_css validated.
    add_test(NAME KHANARY_CSS_GRAMMAR_TEST
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/grammar/mini_transpilers.py" css-check)
    # SDDS/FoldMap transport conformance: direct-seek offsets, split tensors,
    # shard-local duplicate rejection, and physical fold naming for n_layers.
    add_test(NAME KHANARY_SDDS_MANIFEST_TEST
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/check_sdds_manifest_conformance.py")
    # Runtime schema stack gate: enforce mode + declared hash anchors over
    # cycle-hash/fold-runtime and phase driver artifacts.
    add_test(NAME KHANARY_RUNTIME_SCHEMA_STACK_TEST
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/check_runtime_schema_stack.py")
    # Source/contracts vs generated artifacts boundary gate.
    add_test(NAME KHANARY_BOUNDARY_SPLIT_TEST
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/check_boundary_split.py")
endif()

# ChristoffelSymbols (1,2) tensor: conformal metric from entropy gradient,
# torsion-free check, specific Gamma values, flat-limit.  Header-only.
add_executable(KHANARY_CHRISTOFFEL_TEST src/tests/test_christoffel.cpp
    src/kuhul/phase_engine.cpp src/util/json.cpp src/util/sha256.cpp)
target_include_directories(KHANARY_CHRISTOFFEL_TEST PRIVATE src)
add_test(NAME KHANARY_CHRISTOFFEL_TEST COMMAND KHANARY_CHRISTOFFEL_TEST)

# SVG-Tensor loader: k:* attribute scanner, Q/K/V vector parsing, node/edge classification.
# Header-only in io/; no GPU or Factory deps.
add_executable(KHANARY_SVG_TENSOR_TEST src/tests/test_svg_tensor.cpp
    src/io/svg_tensor.cpp src/util/json.cpp src/util/sha256.cpp)
target_include_directories(KHANARY_SVG_TENSOR_TEST PRIVATE src)
add_test(NAME KHANARY_SVG_TENSOR_TEST COMMAND KHANARY_SVG_TENSOR_TEST)

# STBMap binary format: Q/K/V section writer/reader, CRC32 validation, section types.
# Depends on sdds_format for sdds_crc32.
add_executable(KHANARY_STB_MAP_TEST src/tests/test_stb_map.cpp
    src/gpu/stb_map.cpp src/gpu/sdds_format.cpp
    src/util/json.cpp src/util/sha256.cpp)
target_include_directories(KHANARY_STB_MAP_TEST PRIVATE src)
add_test(NAME KHANARY_STB_MAP_TEST COMMAND KHANARY_STB_MAP_TEST)

# SVG-Tensor → STBMap + FieldNode/FieldEdge vector → STBMap converters.
add_executable(KHANARY_STB_MAP_CONVERT_TEST src/tests/test_stb_map_convert.cpp
    src/io/stb_map_convert.cpp src/io/svg_tensor.cpp
    src/gpu/stb_map.cpp src/gpu/sdds_format.cpp
    src/util/json.cpp src/util/sha256.cpp)
target_include_directories(KHANARY_STB_MAP_CONVERT_TEST PRIVATE src)
add_test(NAME KHANARY_STB_MAP_CONVERT_TEST COMMAND KHANARY_STB_MAP_CONVERT_TEST)

add_executable(KHANARY_KLSL_TEST src/tests/test_klsl.cpp src/klsl/klsl_lexer.cpp
    src/klsl/klsl_parser.cpp src/klsl/klsl_ir.cpp src/util/json.cpp src/util/sha256.cpp)
target_include_directories(KHANARY_KLSL_TEST PRIVATE src)
add_test(NAME KHANARY_KLSL_TEST COMMAND KHANARY_KLSL_TEST)

add_executable(KHANARY_WGSL_CODEGEN_TEST src/tests/test_wgsl_codegen.cpp
    src/klsl/klsl_lexer.cpp src/klsl/klsl_parser.cpp src/klsl/klsl_ir.cpp
    src/klsl/wgsl_codegen.cpp src/util/json.cpp src/util/sha256.cpp)
target_include_directories(KHANARY_WGSL_CODEGEN_TEST PRIVATE src)
add_test(NAME KHANARY_WGSL_CODEGEN_TEST COMMAND KHANARY_WGSL_CODEGEN_TEST)

# Key-grammar control flow: sugar -> phase-bound plan -> executable XCFE node.
add_executable(KHANARY_CONTROLFLOW_TEST src/tests/test_control_flow.cpp
    src/micronaut/control_flow.cpp src/micronaut/mx2lm-ngram.cpp
    src/micronaut/eliza_operators.cpp src/micronaut/relation_isa.cpp
    src/kuhul/phase_engine.cpp src/kuhul/xcfe_executor.cpp
    src/util/json.cpp src/util/sha256.cpp)
target_include_directories(KHANARY_CONTROLFLOW_TEST PRIVATE src)
add_test(NAME KHANARY_CONTROLFLOW_TEST COMMAND KHANARY_CONTROLFLOW_TEST)

# Adaptive salutation / greeting-bank policy — no GPU or Factory deps.
add_executable(KHANARY_GREETING_TEST src/tests/test_greeting.cpp
    src/micronaut/greeting_style.cpp src/util/json.cpp src/util/sha256.cpp)
target_include_directories(KHANARY_GREETING_TEST PRIVATE src)
add_test(NAME KHANARY_GREETING_TEST COMMAND KHANARY_GREETING_TEST
    "${CMAKE_SOURCE_DIR}/data/grammar/greeting-bank.json")
add_test(NAME KHANARY_ADVISOR_BOOTSTRAP_TEST
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/grammar/mini_transpilers.py" advisor-check)
add_test(NAME KHANARY_PLAN_BOOTSTRAP_TEST
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/grammar/mini_transpilers.py" plan-check)
add_test(NAME KHANARY_JOKE_BOOTSTRAP_TEST
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/grammar/mini_transpilers.py" joke-check)
set_tests_properties(KHANARY_KLSL_TEST PROPERTIES TIMEOUT 10)
