cmake_minimum_required(VERSION 3.28)
project(KHANARY CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Canonical executable output.  Build trees stay disposable; runtime launchers
# and user-facing scripts should resolve KHANARY executables from repo-local bin/.
# KHANARY_BIN_DIR can be overridden from the cmake command line (e.g. for
# conformance builds on drives where execution is restricted).
if(NOT KHANARY_BIN_DIR)
    set(KHANARY_BIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/bin")
endif()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${KHANARY_BIN_DIR}")
foreach(KHANARY_CFG Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER "${KHANARY_CFG}" KHANARY_CFG_UPPER)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${KHANARY_CFG_UPPER} "${KHANARY_BIN_DIR}")
endforeach()

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
    src/kuhul/micronaut_objective.cpp
    src/kuhul/micronaut_agent.cpp
    src/kuhul/micronaut_intention.cpp
    src/kuhul/kuhul_vm.cpp
    src/kuhul/fold_gate.cpp
    src/kuhul/fold_algebra.cpp
    src/kuhul/phase_driver_lock.cpp
    src/kuhul/xcfe_executor.cpp
    src/kuhul/gram_store.cpp
    src/kuhul/semantic_accumulator.cpp
    src/kuhul/semantic_grammar.cpp
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
    src/gpu/fold_score_dispatch.cpp
    src/gpu/svg_compute_runtime.cpp
    src/gpu/dds_loader.cpp
    src/gpu/sdds_format.cpp
    src/gpu/hot_swap.cpp
    src/inference/tokenizer.cpp
    src/inference/moe_router.cpp
    src/inference/inference_engine.cpp
    src/inference/semantic_attention.cpp
    src/inference/mu_attention.cpp
    src/graphscript/attention_algebra.cpp
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
    src/micronaut/micronaut_coder.cpp
    src/micronaut/coder_pipeline.cpp
    src/micronaut/micronaut_program_runner.cpp
    src/micronaut/micronaut_todo_creator.cpp
    src/micronaut/cfpc.cpp
    src/micronaut/cfpc_intent_resolver.cpp
    src/micronaut/tools_def.cpp
    src/micronaut/tool_registry.cpp
    src/micronaut/control_flow.cpp
    src/micronaut/mou_router.cpp
    src/micronaut/micronaut_transpiler.cpp
    src/micronaut/memoryfield.cpp
    src/micronaut/wcr.cpp
    src/micronaut/cheese.cpp
    src/micronaut/sheogorath.cpp
    src/micronaut/kuhul_vocab.cpp
    src/micronaut/comparison.cpp
    src/micronaut/neighborhood_selector.cpp
    src/micronaut/attention_candidate.cpp
    src/micronaut/question_generator.cpp
    src/micronaut/pne_schema.cpp
    src/micronaut/elaboration_pattern.cpp
    src/micronaut/comparison_resolver.cpp
    src/micronaut/event_role_extractor.cpp
    src/micronaut/pattern_neighborhood_expander.cpp
    src/micronaut/semantic_attention_fold.cpp
    src/micronaut/question_engine.cpp
    src/micronaut/context_fluency.cpp
    src/kuhul/question_selector.cpp
    src/kuhul/question_graph.cpp
    src/micronaut/wildcard_micronaut.cpp
    src/micronaut/intent_resolver.cpp
    src/world/world_state.cpp
    src/world/field_dag.cpp
    src/world/gml_parser.cpp
    src/world/gyroscope.cpp
    src/world/vector_projection.cpp
    src/micronaut/factory.cpp
    src/graphscript/graphscript.cpp
    src/graphscript/graph_algebra.cpp
    src/kuhul/algebra_contract.cpp
    src/graphscript/attention_algebra.cpp
    src/field2d/field2d.cpp
    src/memory/perception.cpp
    src/memory/confidence.cpp
    src/memory/semantic_graph.cpp
    src/memory/scoring.cpp
    src/io/json_rpc_server.cpp
    src/io/chat_completion.cpp
    src/io/mcp_server.cpp
    src/io/mcp_client.cpp
    src/io/code_parser.cpp
    src/io/graphscript_adapter.cpp
    src/io/jrom.cpp
    src/io/kql.cpp
    src/io/projection.cpp
    src/io/runtime_attention.cpp
    src/io/request_router.cpp
    src/io/semantic_driver.cpp
    src/io/aiml_meta.cpp
    src/io/micronaut_grammar.cpp
    src/io/micronaut_tensor_ref.cpp
    src/io/svg_tensor.cpp
    src/io/stb_map_convert.cpp
    src/gpu/stb_map.cpp
    src/util/sha256.cpp
    src/util/json.cpp
    src/util/schema.cpp
    src/util/canonical_hash.cpp
    src/util/lexicon.cpp
    src/util/thread_pool.cpp
    src/llm/llama_infer.cpp
    src/llm/model_swapper.cpp
)

target_include_directories(KHANARY PRIVATE src)

target_link_libraries(KHANARY PRIVATE
    d3d11
    dxgi
    d3dcompiler
    dxguid
    Ws2_32
    Shlwapi
    kh_llama
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
    src/util/lexicon.cpp
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


# ResearchRun smoke test — requires curl.exe on PATH + internet access.
# Run: bin/KHANARY_RESEARCH_TEST.exe             (all tests, live network)
#      bin/KHANARY_RESEARCH_TEST.exe --no-network (url_encode only, no network)
add_executable(KHANARY_RESEARCH_TEST
    src/tests/test_research_run.cpp
    src/host/research_run.cpp
)
target_include_directories(KHANARY_RESEARCH_TEST PRIVATE src src/host)

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

add_executable(KHANARY_AIML_PHASE_BANK_TEST
    src/tests/test_aiml_phase_bank.cpp
    src/micronaut/aiml_engine.cpp
    src/micronaut/eliza_operators.cpp
    src/micronaut/relation_isa.cpp
    src/util/json.cpp
    src/util/sha256.cpp
)
target_include_directories(KHANARY_AIML_PHASE_BANK_TEST PRIVATE src)

add_executable(KHANARY_QUESTION_SELECTOR_TEST
    src/tests/test_question_selector.cpp
    src/kuhul/question_selector.cpp
    src/kuhul/gram_store.cpp
    src/micronaut/aiml_engine.cpp
    src/micronaut/eliza_operators.cpp
    src/micronaut/relation_isa.cpp
    src/util/json.cpp
    src/util/sha256.cpp
)
target_include_directories(KHANARY_QUESTION_SELECTOR_TEST PRIVATE src)

add_executable(KHANARY_QUESTION_GRAPH_TEST
    src/tests/test_question_graph.cpp
    src/kuhul/question_graph.cpp
    src/kuhul/phase_engine.cpp
    src/util/json.cpp
    src/util/sha256.cpp
)
target_include_directories(KHANARY_QUESTION_GRAPH_TEST PRIVATE src)

add_executable(KHANARY_WILDCARD_MICRONAUT_TEST
    src/tests/test_wildcard_micronaut.cpp
    src/micronaut/wildcard_micronaut.cpp
    src/micronaut/memoryfield.cpp
    src/micronaut/aiml_engine.cpp
    src/micronaut/eliza_operators.cpp
    src/micronaut/relation_isa.cpp
    src/micronaut/mx2lm-ngram.cpp
    src/micronaut/control_flow.cpp
    src/micronaut/wcr.cpp
    src/kuhul/question_graph.cpp
    src/kuhul/gram_store.cpp
    src/kuhul/phase_engine.cpp
    src/util/lexicon.cpp
    src/util/json.cpp
    src/util/sha256.cpp
)
target_include_directories(KHANARY_WILDCARD_MICRONAUT_TEST PRIVATE src)

add_executable(KHANARY_ATTENTION_TEST
    src/tests/test_runtime_attention.cpp
    src/io/runtime_attention.cpp
)
target_include_directories(KHANARY_ATTENTION_TEST PRIVATE src)

add_executable(KHANARY_ROUTER_TEST
    src/tests/test_router_policy.cpp
    src/io/request_router.cpp
    src/io/micronaut_grammar.cpp
    src/io/micronaut_tensor_ref.cpp
    src/io/runtime_attention.cpp
    src/io/jrom.cpp
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
    src/kuhul/state_algebra.cpp
    src/kuhul/algebra_contract.cpp
    src/util/lexicon.cpp
    src/util/json.cpp
    src/util/sha256.cpp
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
    src/util/lexicon.cpp
    src/util/json.cpp
    src/util/sha256.cpp
)
target_include_directories(KHANARY_NGRAM_TEST PRIVATE src)

add_executable(KHANARY_COMPARISON_TEST
    src/tests/test_comparison_grammar.cpp
    src/micronaut/comparison.cpp
    src/micronaut/aiml_engine.cpp
    src/micronaut/eliza_operators.cpp
    src/micronaut/relation_isa.cpp
    src/micronaut/mx2lm-ngram.cpp
    src/micronaut/control_flow.cpp
    src/kuhul/phase_engine.cpp
    src/util/lexicon.cpp
    src/util/json.cpp
    src/util/sha256.cpp
)
target_include_directories(KHANARY_COMPARISON_TEST PRIVATE src)

add_executable(KHANARY_NEIGHBORHOOD_TEST
    src/tests/test_neighborhood_selector.cpp
    src/micronaut/neighborhood_selector.cpp
    src/micronaut/comparison.cpp
    src/micronaut/aiml_engine.cpp
    src/micronaut/eliza_operators.cpp
    src/micronaut/relation_isa.cpp
    src/micronaut/mx2lm-ngram.cpp
    src/micronaut/control_flow.cpp
    src/micronaut/memoryfield.cpp
    src/micronaut/wcr.cpp
    src/kuhul/gram_store.cpp
    src/kuhul/phase_engine.cpp
    src/util/lexicon.cpp
    src/util/json.cpp
    src/util/sha256.cpp
)
target_include_directories(KHANARY_NEIGHBORHOOD_TEST PRIVATE src)

add_executable(KHANARY_ATTN_CANDIDATE_TEST
    src/tests/test_attention_candidate.cpp
    src/micronaut/attention_candidate.cpp
    src/micronaut/neighborhood_selector.cpp
    src/micronaut/comparison.cpp
    src/micronaut/aiml_engine.cpp
    src/micronaut/eliza_operators.cpp
    src/micronaut/relation_isa.cpp
    src/micronaut/mx2lm-ngram.cpp
    src/micronaut/control_flow.cpp
    src/micronaut/memoryfield.cpp
    src/micronaut/wcr.cpp
    src/kuhul/gram_store.cpp
    src/kuhul/phase_engine.cpp
    src/util/lexicon.cpp
    src/util/json.cpp
    src/util/sha256.cpp
)
target_include_directories(KHANARY_ATTN_CANDIDATE_TEST PRIVATE src)

# FoldScoreDispatch: CPU/GPU parity for kuhul_fold_score.hlsl.
# Path A (dispatch_score_only): S_i = a_i * e_i using CPU-derived attention scores.
# Path B (dispatch_with_softmax): CSSegmentedSoftmax + scoring kernel parity.
# Requires D3D11; all GPU cases skip gracefully when no device is available.
add_executable(KHANARY_FOLD_SCORE_DISPATCH_TEST
    src/tests/test_fold_score_dispatch.cpp
    src/gpu/fold_score_dispatch.cpp
    src/gpu/gram_gpu_scorer.cpp
    src/gpu/tensor_buffer.cpp
    src/gpu/d3d11_device.cpp
    src/gpu/stb_map.cpp
    src/gpu/sdds_format.cpp
    src/micronaut/attention_candidate.cpp
    src/micronaut/neighborhood_selector.cpp
    src/micronaut/comparison.cpp
    src/micronaut/aiml_engine.cpp
    src/micronaut/eliza_operators.cpp
    src/micronaut/relation_isa.cpp
    src/micronaut/mx2lm-ngram.cpp
    src/micronaut/control_flow.cpp
    src/micronaut/memoryfield.cpp
    src/micronaut/wcr.cpp
    src/kuhul/gram_store.cpp
    src/kuhul/phase_engine.cpp
    src/util/lexicon.cpp
    src/util/json.cpp
    src/util/sha256.cpp
)
target_include_directories(KHANARY_FOLD_SCORE_DISPATCH_TEST PRIVATE src)
target_link_libraries(KHANARY_FOLD_SCORE_DISPATCH_TEST PRIVATE d3d11 dxgi dxguid d3dcompiler)

add_executable(KHANARY_SEMANTIC_ACCUM_TEST
    src/tests/test_semantic_accumulator.cpp
    src/kuhul/semantic_accumulator.cpp
    src/kuhul/phase_engine.cpp
    src/util/json.cpp
    src/util/sha256.cpp
)
target_include_directories(KHANARY_SEMANTIC_ACCUM_TEST PRIVATE src)

add_executable(KHANARY_QUESTION_ENGINE_TEST
    src/tests/test_question_engine.cpp
    src/kuhul/question_graph.cpp
    src/micronaut/question_engine.cpp
    src/micronaut/question_generator.cpp
    src/micronaut/elaboration_pattern.cpp
    src/micronaut/event_role_extractor.cpp
    src/micronaut/semantic_attention_fold.cpp
    src/gpu/fold_score_dispatch.cpp
    src/gpu/gram_gpu_scorer.cpp
    src/gpu/tensor_buffer.cpp
    src/gpu/d3d11_device.cpp
    src/gpu/stb_map.cpp
    src/gpu/sdds_format.cpp
    src/micronaut/attention_candidate.cpp
    src/micronaut/neighborhood_selector.cpp
    src/micronaut/comparison.cpp
    src/micronaut/aiml_engine.cpp
    src/micronaut/eliza_operators.cpp
    src/micronaut/relation_isa.cpp
    src/micronaut/mx2lm-ngram.cpp
    src/micronaut/control_flow.cpp
    src/micronaut/memoryfield.cpp
    src/micronaut/wcr.cpp
    src/kuhul/gram_store.cpp
    src/kuhul/semantic_accumulator.cpp
    src/kuhul/phase_engine.cpp
    src/util/lexicon.cpp
    src/util/json.cpp
    src/util/sha256.cpp
)
target_include_directories(KHANARY_QUESTION_ENGINE_TEST PRIVATE src)
target_link_libraries(KHANARY_QUESTION_ENGINE_TEST PRIVATE d3d11 dxgi dxguid d3dcompiler)

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
    NAME KHANARY_AIML_PHASE_BANK_TEST
    COMMAND $<TARGET_FILE:KHANARY_AIML_PHASE_BANK_TEST>
            "${CMAKE_SOURCE_DIR}/data")
add_test(
    NAME KHANARY_QUESTION_SELECTOR_TEST
    COMMAND $<TARGET_FILE:KHANARY_QUESTION_SELECTOR_TEST>
            "${CMAKE_SOURCE_DIR}/data")
add_test(
    NAME KHANARY_QUESTION_GRAPH_TEST
    COMMAND $<TARGET_FILE:KHANARY_QUESTION_GRAPH_TEST>)
add_test(
    NAME KHANARY_WILDCARD_MICRONAUT_TEST
    COMMAND $<TARGET_FILE:KHANARY_WILDCARD_MICRONAUT_TEST>)
add_test(
    NAME KHANARY_ATTENTION_TEST
    COMMAND $<TARGET_FILE:KHANARY_ATTENTION_TEST>)
add_test(
    NAME KHANARY_ROUTER_TEST
    COMMAND $<TARGET_FILE:KHANARY_ROUTER_TEST>)
add_test(
    NAME KHANARY_NGRAM_TEST
    COMMAND $<TARGET_FILE:KHANARY_NGRAM_TEST>)
add_test(
    NAME KHANARY_COMPARISON_TEST
    COMMAND $<TARGET_FILE:KHANARY_COMPARISON_TEST>)
add_test(
    NAME KHANARY_NEIGHBORHOOD_TEST
    COMMAND $<TARGET_FILE:KHANARY_NEIGHBORHOOD_TEST>)
add_test(
    NAME KHANARY_ATTN_CANDIDATE_TEST
    COMMAND $<TARGET_FILE:KHANARY_ATTN_CANDIDATE_TEST>)
add_test(
    NAME KHANARY_FOLD_SCORE_DISPATCH_TEST
    COMMAND $<TARGET_FILE:KHANARY_FOLD_SCORE_DISPATCH_TEST>)
add_test(
    NAME KHANARY_SEMANTIC_ACCUM_TEST
    COMMAND $<TARGET_FILE:KHANARY_SEMANTIC_ACCUM_TEST>)
add_test(
    NAME KHANARY_QUESTION_ENGINE_TEST
    COMMAND $<TARGET_FILE:KHANARY_QUESTION_ENGINE_TEST>)

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

add_executable(KHANARY_NEIGHBORHOOD_SHADOW_TEST
    src/tests/test_neighborhood_shadow.cpp
    src/micronaut/track_loader.cpp
    src/micronaut/micronaut_registry.cpp
    src/micronaut/relation_isa.cpp
    src/micronaut/gravity_field.cpp
    src/kuhul/phase_engine.cpp
    src/util/json.cpp src/util/schema.cpp src/util/sha256.cpp)
target_include_directories(KHANARY_NEIGHBORHOOD_SHADOW_TEST PRIVATE src)
add_test(NAME KHANARY_NEIGHBORHOOD_SHADOW_TEST COMMAND KHANARY_NEIGHBORHOOD_SHADOW_TEST
    "${CMAKE_SOURCE_DIR}/data")

add_executable(KHANARY_NEIGHBORHOOD_STAGE3_TEST
    src/tests/test_neighborhood_stage3.cpp
    src/micronaut/track_loader.cpp
    src/micronaut/micronaut_registry.cpp
    src/micronaut/relation_isa.cpp
    src/micronaut/gravity_field.cpp
    src/kuhul/phase_engine.cpp
    src/util/json.cpp src/util/schema.cpp src/util/sha256.cpp)
target_include_directories(KHANARY_NEIGHBORHOOD_STAGE3_TEST PRIVATE src)
add_test(NAME KHANARY_NEIGHBORHOOD_STAGE3_TEST COMMAND KHANARY_NEIGHBORHOOD_STAGE3_TEST
    "${CMAKE_SOURCE_DIR}/data")

add_executable(KHANARY_CFPC_TEST
    src/tests/test_cfpc.cpp
    src/micronaut/cfpc.cpp
    src/micronaut/track_loader.cpp
    src/micronaut/micronaut_registry.cpp
    src/micronaut/relation_isa.cpp
    src/micronaut/gravity_field.cpp
    src/kuhul/phase_engine.cpp
    src/util/json.cpp src/util/schema.cpp src/util/sha256.cpp)
target_include_directories(KHANARY_CFPC_TEST PRIVATE src)
add_test(NAME KHANARY_CFPC_TEST COMMAND KHANARY_CFPC_TEST
    "${CMAKE_SOURCE_DIR}/data")

add_executable(KHANARY_CFPC_INTENT_TEST
    src/tests/test_cfpc_intent.cpp
    src/micronaut/cfpc_intent_resolver.cpp
    src/micronaut/cfpc.cpp
    src/micronaut/track_loader.cpp
    src/micronaut/micronaut_registry.cpp
    src/micronaut/relation_isa.cpp
    src/micronaut/gravity_field.cpp
    src/kuhul/phase_engine.cpp
    src/util/json.cpp src/util/schema.cpp src/util/sha256.cpp)
target_include_directories(KHANARY_CFPC_INTENT_TEST PRIVATE src)
add_test(NAME KHANARY_CFPC_INTENT_TEST COMMAND KHANARY_CFPC_INTENT_TEST
    "${CMAKE_SOURCE_DIR}/data")

add_executable(KHANARY_CFPC_INTENT_STAGE1_TEST
    src/tests/test_cfpc_intent_stage1.cpp
    src/micronaut/cfpc_intent_resolver.cpp
    src/micronaut/cfpc.cpp
    src/micronaut/micronaut_registry.cpp
    src/micronaut/relation_isa.cpp
    src/micronaut/gravity_field.cpp
    src/kuhul/phase_engine.cpp
    src/util/json.cpp src/util/schema.cpp src/util/sha256.cpp)
target_include_directories(KHANARY_CFPC_INTENT_STAGE1_TEST PRIVATE src)
add_test(NAME KHANARY_CFPC_INTENT_STAGE1_TEST COMMAND KHANARY_CFPC_INTENT_STAGE1_TEST)

add_executable(KHANARY_CFPC_INTENT_STAGE2_TEST
    src/tests/test_cfpc_intent_stage2.cpp
    src/micronaut/cfpc_intent_resolver.cpp
    src/micronaut/cfpc.cpp
    src/micronaut/micronaut_registry.cpp
    src/micronaut/relation_isa.cpp
    src/micronaut/gravity_field.cpp
    src/kuhul/phase_engine.cpp
    src/util/json.cpp src/util/schema.cpp src/util/sha256.cpp)
target_include_directories(KHANARY_CFPC_INTENT_STAGE2_TEST PRIVATE src)
add_test(NAME KHANARY_CFPC_INTENT_STAGE2_TEST COMMAND KHANARY_CFPC_INTENT_STAGE2_TEST)

add_executable(KHANARY_CFPC_INTENT_STAGE3_TEST
    src/tests/test_cfpc_intent_stage3.cpp
    src/micronaut/cfpc_intent_resolver.cpp
    src/micronaut/cfpc.cpp
    src/micronaut/micronaut_registry.cpp
    src/micronaut/relation_isa.cpp
    src/micronaut/gravity_field.cpp
    src/kuhul/phase_engine.cpp
    src/util/json.cpp src/util/schema.cpp src/util/sha256.cpp)
target_include_directories(KHANARY_CFPC_INTENT_STAGE3_TEST PRIVATE src)
add_test(NAME KHANARY_CFPC_INTENT_STAGE3_TEST COMMAND KHANARY_CFPC_INTENT_STAGE3_TEST)

# Typed binding contract: BindingFrame, TypedBinding, BindingScopeConflict.
# Header-only types (cfpc_binding.h); no resolver, registry, or GPU deps.
# Tests conformance #25-28 (scope rules, parallel return, provenance retention).
add_executable(KHANARY_CFPC_BINDING_TEST
    src/tests/test_cfpc_binding.cpp
    src/util/json.cpp src/util/sha256.cpp)
target_include_directories(KHANARY_CFPC_BINDING_TEST PRIVATE src)
add_test(NAME KHANARY_CFPC_BINDING_TEST COMMAND KHANARY_CFPC_BINDING_TEST)

add_executable(KHANARY_CANONICAL_TEST src/tests/test_canonical.cpp
    src/util/json.cpp src/util/sha256.cpp src/util/canonical_hash.cpp)
target_include_directories(KHANARY_CANONICAL_TEST PRIVATE src)
add_test(NAME KHANARY_CANONICAL_TEST COMMAND KHANARY_CANONICAL_TEST "${CMAKE_SOURCE_DIR}/data/conformance/canonical.json")

add_executable(KHANARY_SDDS_TEST src/tests/test_sdds.cpp src/gpu/sdds_format.cpp
    src/util/json.cpp src/util/sha256.cpp)
target_include_directories(KHANARY_SDDS_TEST PRIVATE src)
add_test(NAME KHANARY_SDDS_TEST COMMAND KHANARY_SDDS_TEST)

add_executable(KHANARY_TENSOR_STORAGE_TEST
    src/tests/test_tensor_storage_adapter.cpp
    src/gpu/tensor_storage_adapter.cpp
    src/gpu/sdds_format.cpp
    src/util/json.cpp src/util/sha256.cpp)
target_include_directories(KHANARY_TENSOR_STORAGE_TEST PRIVATE src)
add_test(NAME KHANARY_TENSOR_STORAGE_TEST COMMAND KHANARY_TENSOR_STORAGE_TEST)

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

add_executable(KHANARY_CONVERGENCE_TEST src/tests/test_convergence.cpp
    src/memory/convergence.cpp src/util/json.cpp src/util/sha256.cpp)
target_include_directories(KHANARY_CONVERGENCE_TEST PRIVATE src)
add_test(NAME KHANARY_CONVERGENCE_TEST COMMAND KHANARY_CONVERGENCE_TEST)

add_executable(KHANARY_FOLD_FIELD_TEST src/tests/test_fold_field.cpp
    src/memory/fold_field.cpp src/kuhul/phase_engine.cpp src/util/json.cpp src/util/sha256.cpp)
target_include_directories(KHANARY_FOLD_FIELD_TEST PRIVATE src)
add_test(NAME KHANARY_FOLD_FIELD_TEST COMMAND KHANARY_FOLD_FIELD_TEST)

add_executable(KHANARY_PHASE_DRIVER_LOCK_TEST src/tests/test_phase_driver_lock.cpp
    src/kuhul/phase_driver_lock.cpp src/kuhul/phase_engine.cpp src/util/json.cpp src/util/sha256.cpp)
target_include_directories(KHANARY_PHASE_DRIVER_LOCK_TEST PRIVATE src)
add_test(NAME KHANARY_PHASE_DRIVER_LOCK_TEST
    COMMAND KHANARY_PHASE_DRIVER_LOCK_TEST "${CMAKE_SOURCE_DIR}" "${CMAKE_BINARY_DIR}")

find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    add_test(NAME KHANARY_SCHEMA_TEST COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/check_contracts.py")
    # Public API gateway: api.kuhul.dev:443 is the reachable edge contract;
    # the local gateway remains a projection and cannot bypass Ch'en/XCFE/Xul.
    add_test(NAME KHANARY_API_GATEWAY_TEST
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/check_api_gateway.py")
    # Apps host lane: src/host owns user-facing app/playground surfaces while
    # MCP remains a separate model/tool protocol lane.
    add_test(NAME KHANARY_APP_HOST_TEST
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/check_app_host.py")
    # Driver bundle host map: every artifact names its owning host process, the
    # managed deps.json graphs resolve, and the vendored hashes match.
    add_test(NAME KHANARY_DRIVER_MANIFEST_TEST
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/check_driver_manifest.py")
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
    # XCFE Node Graph -> KXML is a trusted lowering boundary. KXML compile is
    # only JSON-to-CBOR encoding and cannot infer semantic admission.
    add_test(NAME KHANARY_XCFE_KXML_LOWERING_TEST
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/check_xcfe_kxml_lowering.py")
    # Fold Space navigation: full runtime positions over fold handle, phase,
    # track, fold/unfold extent, optional competence, and mutable state.
    add_test(NAME KHANARY_FOLD_SPACE_TEST
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/check_fold_space.py")
    # Fold Event JSON Schema: events log proposed/observed Fold Space movement;
    # XCFE owns p_next = delta_XCFE(p_t, e_t, Pi) transition authority.
    add_test(NAME KHANARY_FOLD_EVENT_TEST
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/check_fold_event.py")
    # Fold Graph recoverable contraction and orientation: Handle is an
    # addressable contracted graph state; traversal must satisfy semantic edge
    # orientation plus C6 phase-forward law.
    add_test(NAME KHANARY_FOLD_GRAPH_TEST
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/check_fold_graph.py")
    # Fold Lattice product graph: phase theta and competence sigma stay
    # orthogonal, with metrics recommending movement but XCFE retaining authority.
    add_test(NAME KHANARY_FOLD_LATTICE_TEST
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/check_fold_lattice.py")
    # Fold Kernel lifecycle: state-dependent JSON kernel records grow from
    # learned n-grams through binding/evidence/policy into XCFE-owned routes.
    add_test(NAME KHANARY_FOLD_KERNEL_TEST
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/check_fold_kernel.py")
    # Fold Function grammar: text math becomes graph-shaped projection syntax;
    # K'ayab'/Kumk'u compiler/GPU phases stay outside C6 semantic folds.
    add_test(NAME KHANARY_FOLD_FUNCTION_TEST
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/check_fold_function.py")
    # The capability-gap artifact must stay current with the operator table and
    # the tracks' declared capabilities.
    add_test(NAME KHANARY_CAPABILITY_GAP_TEST
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/grammar/mini_transpilers.py" capability-gap --check)
    # Code-grammar artifacts: <Source, Freshness, Shape> per artifact, with a
    # missing canonical source reported as UNVERIFIED rather than invented.
    add_test(NAME KHANARY_CODE_GRAMMAR_TEST
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/grammar/mini_transpilers.py" code-grammar-check)
    # Semantic-grammar artifact: §24/§25/§26 closed sets + §73 attributed
    # lexical grounding, compiled from the canonical EBNF by the §71 target
    # compiler pass; <Source, Freshness, Shape> triad like the code-grammar lane.
    add_test(NAME KHANARY_SEMANTIC_GRAMMAR_TEST
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/check_semantic_grammar.py")
    # Graphscript runtime-grammar spec surface (sections, invariants, §11
    # codegen tables, §12 terminals) + its compiled lexical companion's tokens.
    add_test(NAME KHANARY_GRAPHSCRIPT_GRAMMAR_TEST
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/check_graphscript_grammar.py")
    # Field2d spec surface: the consolidated π-phase grammar (field/tensor/
    # phase/fold/lane/graph + the descent-lane extension).
    add_test(NAME KHANARY_FIELD2D_GRAMMAR_TEST
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/check_field2d_grammar.py")
    # 2D key-graph execution field: phase-qualified graph programs plus
    # projection-only bias/buffer/provider/kernel dispatch contracts.
    add_test(NAME KHANARY_KEY_GRAPH_EXECUTION_FIELD_TEST
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/check_2d_key_graph_execution_field.py")
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
    add_test(NAME KHANARY_MICRONAUT_PROGRAM_TEST
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/check_micronaut_program.py")
    add_test(NAME KHANARY_NGRAM_KEY_GRAMMAR_TEST
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/check_ngram_key_grammar.py")
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
    src/util/lexicon.cpp
    src/util/json.cpp src/util/sha256.cpp)
target_include_directories(KHANARY_CONTROLFLOW_TEST PRIVATE src)
add_test(NAME KHANARY_CONTROLFLOW_TEST COMMAND KHANARY_CONTROLFLOW_TEST)

# KuhulVM safety: GAP-101 required sandbox at gram-op boundaries,
# GAP-102 per-execution fuel limits, GAP-109 XCFE condition tokenizer.
add_executable(KHANARY_KUHUL_VM_TEST src/tests/test_kuhul_vm.cpp
    src/kuhul/kuhul_vm.cpp src/kuhul/fold_gate.cpp src/kuhul/fold_algebra.cpp
    src/kuhul/algebra_contract.cpp
    src/kuhul/phase_engine.cpp src/kuhul/fold_engine.cpp
    src/kuhul/xcfe_executor.cpp src/kuhul/gram_store.cpp
    src/kuhul/phase_driver_lock.cpp src/kuhul/semantic_grammar.cpp
    src/micronaut/sandbox_inspector.cpp src/micronaut/aiml_engine.cpp
    src/micronaut/eliza_operators.cpp src/micronaut/relation_isa.cpp
    src/util/json.cpp src/util/sha256.cpp)
target_include_directories(KHANARY_KUHUL_VM_TEST PRIVATE src)
add_test(NAME KHANARY_KUHUL_VM_TEST COMMAND KHANARY_KUHUL_VM_TEST)

# KVCache: GAP-094 typed capacity bounds; GAP-095 prompt lifecycle + bounded
# attention-score log. Skips explicitly when no D3D11 device is available.
add_executable(KHANARY_KV_CACHE_TEST src/tests/test_kv_cache.cpp
    src/inference/kv_cache.cpp src/gpu/tensor_buffer.cpp src/gpu/d3d11_device.cpp
    src/util/json.cpp src/util/sha256.cpp)
target_include_directories(KHANARY_KV_CACHE_TEST PRIVATE src)
target_link_libraries(KHANARY_KV_CACHE_TEST PRIVATE d3d11 dxgi dxguid)
add_test(NAME KHANARY_KV_CACHE_TEST COMMAND KHANARY_KV_CACHE_TEST)

# MCP tool contract: GAP-103 gram_write Ch'en authority gate; GAP-104 typed
# memory:write / memory:recall JROM audit events with chain integrity.
add_executable(KHANARY_MCP_POLICY_TEST src/tests/test_mcp_policy.cpp
    src/io/jrom.cpp src/util/json.cpp src/util/sha256.cpp)
target_include_directories(KHANARY_MCP_POLICY_TEST PRIVATE src)
add_test(NAME KHANARY_MCP_POLICY_TEST COMMAND KHANARY_MCP_POLICY_TEST)

# Lorentz transition scoping: GAP-047 prompt/world-generation binding, expiry,
# monotonic sequence. Header-only logic; wired into InferenceEngine.
add_executable(KHANARY_LORENTZ_SCOPE_TEST src/tests/test_lorentz_scope.cpp
    src/util/json.cpp src/util/sha256.cpp)
target_include_directories(KHANARY_LORENTZ_SCOPE_TEST PRIVATE src)
add_test(NAME KHANARY_LORENTZ_SCOPE_TEST COMMAND KHANARY_LORENTZ_SCOPE_TEST)

# Lexical layer: tokenization declared in the grammar's @lex/@profile sections
# (compiled by tools/ebnf_to_code_grammar.py), consumed through
# src/util/lexicon.cpp. Pins the compiled core table to the
# data/grammar/cpp.code-grammar.json lexicon block so EBNF and engine cannot
# drift, and pins each unified lane's historical behavior.
add_executable(KHANARY_LEXICON_TEST src/tests/test_lexicon.cpp
    src/util/lexicon.cpp src/util/json.cpp src/util/sha256.cpp)
target_include_directories(KHANARY_LEXICON_TEST PRIVATE src)
add_test(NAME KHANARY_LEXICON_TEST COMMAND KHANARY_LEXICON_TEST
    "${CMAKE_SOURCE_DIR}")

# Semantic grammar: cue/atom/rung grounding authority compiled from the
# canonical EBNF (§24/§25/§26/§73) — core mirror pinned to the emitted
# artifact; historical grounding behavior pinned for the io consumer.
add_executable(KHANARY_SEMANTIC_TEST src/tests/test_semantic_grammar.cpp
    src/kuhul/semantic_grammar.cpp src/util/json.cpp src/util/sha256.cpp)
target_include_directories(KHANARY_SEMANTIC_TEST PRIVATE src)
add_test(NAME KHANARY_SEMANTIC_TEST COMMAND KHANARY_SEMANTIC_TEST
    "${CMAKE_SOURCE_DIR}")

# GraphScript codegen lane: the §11 normative transformation rules as
# executable C++ — parse (codegen subset) → six f_* emissions + the pass.
add_executable(KHANARY_GRAPHSCRIPT_TEST src/tests/test_graphscript.cpp
    src/graphscript/graphscript.cpp src/util/json.cpp src/util/sha256.cpp)
target_include_directories(KHANARY_GRAPHSCRIPT_TEST PRIVATE src)
add_test(NAME KHANARY_GRAPHSCRIPT_TEST COMMAND KHANARY_GRAPHSCRIPT_TEST)

add_executable(KHANARY_GRAPHSCRIPT_CODEGEN tools/graphscript_codegen.cpp
    src/graphscript/graphscript.cpp)
target_include_directories(KHANARY_GRAPHSCRIPT_CODEGEN PRIVATE src)
add_test(NAME KHANARY_GRAPHSCRIPT_GOLDEN_TEST COMMAND KHANARY_GRAPHSCRIPT_CODEGEN
    --check "${CMAKE_SOURCE_DIR}/tests/graphscript/golden.graphscript"
    "${CMAKE_SOURCE_DIR}/tests/graphscript/expected")

add_executable(KHANARY_GRAPHSCRIPT_ADAPTER_TEST
    src/tests/test_graphscript_adapter.cpp
    src/io/graphscript_adapter.cpp
    src/graphscript/graphscript.cpp
    src/micronaut/wildcard_micronaut.cpp
    src/micronaut/micronaut_registry.cpp
    src/micronaut/gravity_field.cpp
    src/micronaut/relation_isa.cpp
    src/micronaut/memoryfield.cpp
    src/micronaut/mx2lm-ngram.cpp
    src/micronaut/aiml_engine.cpp
    src/micronaut/eliza_operators.cpp
    src/micronaut/control_flow.cpp
    src/kuhul/gram_store.cpp
    src/kuhul/question_graph.cpp
    src/kuhul/phase_engine.cpp
    src/util/lexicon.cpp
    src/util/json.cpp
    src/util/sha256.cpp)
target_include_directories(KHANARY_GRAPHSCRIPT_ADAPTER_TEST PRIVATE src)
add_test(NAME KHANARY_GRAPHSCRIPT_ADAPTER_TEST COMMAND KHANARY_GRAPHSCRIPT_ADAPTER_TEST)

# Micronaut coder lane (CR-1): deterministic code review / todo extraction /
# v6 JSON artifacts; real sha256 via the util lane. The µ-call graph runs on
# the real control plane (micronaut_registry + kuhul phase names) — same
# source set the router test links.
add_executable(KHANARY_CODER_TEST src/tests/test_micronaut_coder.cpp
    src/micronaut/micronaut_coder.cpp
    src/micronaut/coder_pipeline.cpp
    src/micronaut/micronaut_todo_creator.cpp
    src/micronaut/micronaut_registry.cpp
    src/micronaut/gravity_field.cpp
    src/micronaut/relation_isa.cpp
    src/field2d/field2d.cpp
    src/inference/mu_attention.cpp
    src/inference/semantic_attention.cpp
    src/graphscript/attention_algebra.cpp
    src/kuhul/gram_store.cpp
    src/kuhul/phase_engine.cpp
    src/util/json.cpp src/util/sha256.cpp)
target_include_directories(KHANARY_CODER_TEST PRIVATE src)
add_test(NAME KHANARY_CODER_TEST COMMAND KHANARY_CODER_TEST)

# Micronaut coder pipeline layers: toy parser (real source lines) → IR →
# generator + API writer scaffold.
add_executable(KHANARY_CODER_PIPELINE_TEST src/tests/test_coder_pipeline.cpp
    src/micronaut/coder_pipeline.cpp src/util/json.cpp src/util/sha256.cpp)
target_include_directories(KHANARY_CODER_PIPELINE_TEST PRIVATE src)
add_test(NAME KHANARY_CODER_PIPELINE_TEST COMMAND KHANARY_CODER_PIPELINE_TEST)

# TODO ledger creator: marker extraction, priority algebra, v6 ledger JSON,
# escaping dashboards (strict-parse gates via util::json).
add_executable(KHANARY_TODO_CREATOR_TEST src/tests/test_micronaut_todo_creator.cpp
    src/micronaut/micronaut_todo_creator.cpp src/util/json.cpp src/util/sha256.cpp)
target_include_directories(KHANARY_TODO_CREATOR_TEST PRIVATE src)
add_test(NAME KHANARY_TODO_CREATOR_TEST COMMAND KHANARY_TODO_CREATOR_TEST)

# Micronaut program runner: the revised canonical definition as a runtime —
# nested C6 bracket programs with ladder/wheel admission law, µ-call
# execution, Ch'en certification, σ candidates.
add_executable(KHANARY_MICRONAUT_RUNNER_TEST src/tests/test_micronaut_program_runner.cpp
    src/micronaut/micronaut_program_runner.cpp
    src/micronaut/micronaut_registry.cpp
    src/micronaut/gravity_field.cpp
    src/micronaut/relation_isa.cpp
    src/kuhul/phase_engine.cpp
    src/kuhul/gram_store.cpp
    src/inference/semantic_attention.cpp
    src/inference/mu_attention.cpp
    src/graphscript/attention_algebra.cpp
    src/util/json.cpp src/util/sha256.cpp)
target_include_directories(KHANARY_MICRONAUT_RUNNER_TEST PRIVATE src)
add_test(NAME KHANARY_MICRONAUT_RUNNER_TEST COMMAND KHANARY_MICRONAUT_RUNNER_TEST)

# K'UHUL vocabulary/packing parity: CPU set-based jaccard_max vs GPU binary
# intersection + jaccard_max. Covers all seven architectural parity cases.
add_executable(KHANARY_VOCAB_PARITY_TEST
    src/tests/test_kuhul_vocab_parity.cpp
    src/micronaut/kuhul_vocab.cpp
    src/micronaut/memoryfield.cpp
    src/micronaut/wcr.cpp
    src/micronaut/mx2lm-ngram.cpp
    src/micronaut/control_flow.cpp
    src/kuhul/phase_engine.cpp
    src/util/lexicon.cpp
    src/util/json.cpp
    src/util/sha256.cpp)
target_include_directories(KHANARY_VOCAB_PARITY_TEST PRIVATE src)
add_test(NAME KHANARY_VOCAB_PARITY_TEST COMMAND KHANARY_VOCAB_PARITY_TEST)

# π-Phase 2D field lane: field/graph/fold/lane runtime; the descent-lane
# convergence proof (gradient descent = lane rule) is the pivotal check.
add_executable(KHANARY_FIELD2D_TEST src/tests/test_field2d.cpp
    src/field2d/field2d.cpp src/util/json.cpp src/util/sha256.cpp)
target_include_directories(KHANARY_FIELD2D_TEST PRIVATE src)
add_test(NAME KHANARY_FIELD2D_TEST COMMAND KHANARY_FIELD2D_TEST)

# Graph algebra: the graph-safe Map-Filter-Fold pipeline (induced sub-adjacency
# on filter, named invariant throws, telemetry report).
add_executable(KHANARY_GRAPH_ALGEBRA_TEST src/tests/test_graph_algebra.cpp
    src/graphscript/graph_algebra.cpp src/kuhul/algebra_contract.cpp
    src/util/json.cpp src/util/sha256.cpp)
target_include_directories(KHANARY_GRAPH_ALGEBRA_TEST PRIVATE src)
add_test(NAME KHANARY_GRAPH_ALGEBRA_TEST COMMAND KHANARY_GRAPH_ALGEBRA_TEST)

# Algebraic Operator ISA: which algebraic laws each operator obeys, and the
# execution permissions derived from them (regroup / reorder / parallel /
# repeat / invert). Asserts that the compiled-in axiom table matches
# data/grammar/algebra-isa.contracts.json and that the derived permission
# vector matches tools/check_algebra_isa.py on the shared fixtures.
add_executable(KHANARY_ALGEBRA_ISA_TEST src/tests/test_algebra_isa.cpp
    src/kuhul/algebra_contract.cpp src/kuhul/state_algebra.cpp
    src/graphscript/graph_algebra.cpp
    src/util/json.cpp src/util/sha256.cpp)
target_include_directories(KHANARY_ALGEBRA_ISA_TEST PRIVATE src)
add_test(NAME KHANARY_ALGEBRA_ISA_TEST COMMAND KHANARY_ALGEBRA_ISA_TEST
    "${CMAKE_SOURCE_DIR}/data")
add_test(NAME KHANARY_ALGEBRA_ISA_CHECK
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/check_algebra_isa.py")

# K'UHUL Fold Algebra: a fold is a typed state transition F_phi : S -> S, the
# cycle is a composition, and (F, ., FOLD_IDENTITY) is a monoid. Decides
# structural composability (types, phase order, verification) and leaves
# admission to XCFE. Cross-checks the SRAI and AOI registries from both sides.
add_executable(KHANARY_FOLD_ALGEBRA_TEST src/tests/test_fold_algebra.cpp
    src/kuhul/fold_algebra.cpp src/kuhul/algebra_contract.cpp src/kuhul/phase_engine.cpp
    src/util/json.cpp src/util/sha256.cpp)
target_include_directories(KHANARY_FOLD_ALGEBRA_TEST PRIVATE src)
add_test(NAME KHANARY_FOLD_ALGEBRA_TEST COMMAND KHANARY_FOLD_ALGEBRA_TEST
    "${CMAKE_SOURCE_DIR}/data")
add_test(NAME KHANARY_FOLD_ALGEBRA_CHECK
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/check_fold_algebra.py")

# Fold Gate: the composition law as an execution gate. An absent registry leaves
# the gate inactive (never an outage); Observe records the verdict; Enforce
# refuses the plan before the first instruction runs.
add_executable(KHANARY_FOLD_GATE_TEST src/tests/test_fold_gate.cpp
    src/kuhul/kuhul_vm.cpp src/kuhul/fold_gate.cpp src/kuhul/fold_algebra.cpp
    src/kuhul/algebra_contract.cpp src/kuhul/phase_engine.cpp
    src/kuhul/fold_engine.cpp src/kuhul/xcfe_executor.cpp src/kuhul/gram_store.cpp
    src/kuhul/phase_driver_lock.cpp src/kuhul/semantic_grammar.cpp
    src/micronaut/sandbox_inspector.cpp src/micronaut/aiml_engine.cpp
    src/micronaut/eliza_operators.cpp src/micronaut/relation_isa.cpp
    src/util/json.cpp src/util/sha256.cpp)
target_include_directories(KHANARY_FOLD_GATE_TEST PRIVATE src)
add_test(NAME KHANARY_FOLD_GATE_TEST COMMAND KHANARY_FOLD_GATE_TEST
    "${CMAKE_SOURCE_DIR}/data")

# Attention algebra: softmax(QKᵀ/√d)·V decomposed as map/filter/fold; masked
# self-attention over declared graphs; the W·C·cos kernel (WCR) bridge.
add_executable(KHANARY_ATTENTION_ALGEBRA_TEST src/tests/test_attention_algebra.cpp
    src/graphscript/attention_algebra.cpp src/graphscript/graph_algebra.cpp
    src/kuhul/algebra_contract.cpp
    src/inference/mu_attention.cpp src/inference/semantic_attention.cpp
    src/kuhul/gram_store.cpp
    src/micronaut/micronaut_registry.cpp src/micronaut/gravity_field.cpp
    src/micronaut/relation_isa.cpp src/kuhul/phase_engine.cpp
    src/util/json.cpp src/util/sha256.cpp)
target_include_directories(KHANARY_ATTENTION_ALGEBRA_TEST PRIVATE src)
add_test(NAME KHANARY_ATTENTION_ALGEBRA_TEST COMMAND KHANARY_ATTENTION_ALGEBRA_TEST)

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

# Apps host lane executable (optional; emits bin/KHANARY_APP_HOST.exe).
# src/host owns user-facing app/playground surfaces (REST API, MicroXcfe
# projection runner, JROM+CBOR ROM tools) while MCP stays in src/io/mcp_server
# as the model/tool protocol lane. Loopback-only: binds 127.0.0.1, rejects
# non-loopback Host headers, CORS granted to loopback origins only.
add_executable(KHANARY_APP_HOST
    src/host/main.cpp
    src/host/runtime.cpp
    src/kuhul/phase_engine.cpp
    src/kuhul/micronaut_intention.cpp
    src/util/json.cpp
    src/host/json_runtime_api.cpp
    src/host/JsonProgram.cpp
    src/host/JsonUI.cpp
    src/host/JsonApp.cpp
    src/host/server.cpp
    src/host/gpu_sandbox_adapter.cpp
    src/host/api.cpp
    src/host/kxml_lowering.cpp
    src/host/model_catalog.cpp
    src/host/discovery.cpp
    src/host/file_system.cpp
    src/host/research_run.cpp
    src/llm/llama_infer.cpp
    src/util/sha256.cpp)
target_sources(KHANARY_APP_HOST PRIVATE
    src/gpu/d3d11_device.cpp
    src/gpu/compute_dispatch.cpp
    src/gpu/tensor_buffer.cpp
    src/gpu/sdds_format.cpp
    src/klsl/d3dcompile_bridge.cpp)
target_include_directories(KHANARY_APP_HOST PRIVATE src src/host)
target_link_libraries(KHANARY_APP_HOST PRIVATE Ws2_32 kh_llama d3d11 dxgi d3dcompiler)

# Apps host lane unit coverage: ROM write containment (NG-H2), MicroXcfe
# strict execution, SCO registry, ROM roundtrip, tracked-manifest boot.
add_executable(KHANARY_APP_HOST_RUNTIME_TEST src/tests/test_app_host.cpp
    src/host/runtime.cpp
    src/kuhul/micronaut_intention.cpp
    src/kuhul/phase_engine.cpp
    src/util/json.cpp
    src/host/JsonProgram.cpp
    src/host/JsonUI.cpp
    src/host/JsonApp.cpp
    src/host/api.cpp
    src/host/kxml_lowering.cpp
    src/host/model_catalog.cpp
    src/host/file_system.cpp
    src/util/sha256.cpp)
target_include_directories(KHANARY_APP_HOST_RUNTIME_TEST PRIVATE src src/host)
add_test(NAME KHANARY_APP_HOST_RUNTIME_TEST COMMAND KHANARY_APP_HOST_RUNTIME_TEST
    "${CMAKE_SOURCE_DIR}" "${CMAKE_BINARY_DIR}")

# Hardened JSON-RPC server contract: loopback bind + fail-closed policy,
# HTTP status mapping, request-line policy, bounded framing, bearer auth,
# and connection admission — over real loopback sockets.
add_executable(KHANARY_RPC_SERVER_TEST src/tests/test_rpc_server.cpp
    src/io/json_rpc_server.cpp
    src/util/json.cpp src/util/sha256.cpp)
target_include_directories(KHANARY_RPC_SERVER_TEST PRIVATE src)
add_test(NAME KHANARY_RPC_SERVER_TEST COMMAND KHANARY_RPC_SERVER_TEST)
set_tests_properties(KHANARY_RPC_SERVER_TEST PROPERTIES TIMEOUT 60)

# MCP client smoke test: McpClient + ChatClient against a loopback
# JsonRpcServer. Verifies ping, list_tools, call_tool, and transport.
add_executable(KHANARY_MCP_CLIENT_TEST src/tests/test_mcp_client.cpp
    src/io/json_rpc_server.cpp
    src/io/mcp_client.cpp
    src/util/json.cpp src/util/sha256.cpp)
target_include_directories(KHANARY_MCP_CLIENT_TEST PRIVATE src)
add_test(NAME KHANARY_MCP_CLIENT_TEST COMMAND KHANARY_MCP_CLIENT_TEST)
set_tests_properties(KHANARY_MCP_CLIENT_TEST PROPERTIES TIMEOUT 60)

add_executable(KHANARY_MCP_CLIENT_LIVE_TEST src/tests/test_mcp_client_live.cpp
    src/io/mcp_client.cpp
    src/util/json.cpp src/util/sha256.cpp)
target_include_directories(KHANARY_MCP_CLIENT_LIVE_TEST PRIVATE src)
# Not registered with ctest — requires a live KHANARY instance.
# Run manually: .\run.bat bin\KHANARY_MCP_CLIENT_LIVE_TEST.exe

# Factory commit/provenance/durability fixtures (GAP-062/063): authority
# binding, TrackVector transaction hashes, stale/non-monotonic rejection,
# concurrency consistency, persist→reload roundtrip.
add_executable(KHANARY_FACTORY_TEST src/tests/test_factory.cpp
    src/micronaut/factory.cpp
    src/micronaut/micronaut_registry.cpp
    src/micronaut/gravity_field.cpp
    src/micronaut/relation_isa.cpp
    src/micronaut/mou_router.cpp
    src/micronaut/micronaut_transpiler.cpp
    src/kuhul/phase_engine.cpp
    src/inference/moe_router.cpp
    src/memory/perception.cpp
    src/memory/confidence.cpp
    src/memory/scoring.cpp
    src/memory/semantic_graph.cpp
    src/io/runtime_attention.cpp
    src/io/jrom.cpp
    src/util/json.cpp
    src/util/sha256.cpp)
target_include_directories(KHANARY_FACTORY_TEST PRIVATE src)
add_test(NAME KHANARY_FACTORY_TEST COMMAND KHANARY_FACTORY_TEST)
set_tests_properties(KHANARY_FACTORY_TEST PROPERTIES TIMEOUT 60)

# ── llama-haswell in-process integration ─────────────────────────────────────
# Pre-built static libs (Haswell/AVX2/FMA/OpenCL, BUILD_SHARED_LIBS=OFF).
# Headers vendored in vendor/llama/include/ — no E: drive dependency at build time.
find_package(OpenMP)

set(LLAMA_HW "${CMAKE_SOURCE_DIR}/llama-haswell")

add_library(kh_ggml_base       STATIC IMPORTED GLOBAL)
set_target_properties(kh_ggml_base PROPERTIES
    IMPORTED_LOCATION "${LLAMA_HW}/ggml/src/ggml-base.lib")

add_library(kh_ggml_cpu        STATIC IMPORTED GLOBAL)
set_target_properties(kh_ggml_cpu PROPERTIES
    IMPORTED_LOCATION "${LLAMA_HW}/ggml/src/ggml-cpu.lib"
    INTERFACE_LINK_LIBRARIES "kh_ggml_base$<$<BOOL:${OpenMP_FOUND}>:;OpenMP::OpenMP_CXX>")

add_library(kh_ggml_opencl     STATIC IMPORTED GLOBAL)
set_target_properties(kh_ggml_opencl PROPERTIES
    IMPORTED_LOCATION "${LLAMA_HW}/ggml/src/ggml-opencl/ggml-opencl.lib"
    INTERFACE_LINK_LIBRARIES "kh_ggml_base;${LLAMA_HW}/OpenCL.lib")

add_library(kh_ggml            STATIC IMPORTED GLOBAL)
set_target_properties(kh_ggml PROPERTIES
    IMPORTED_LOCATION "${LLAMA_HW}/ggml/src/ggml.lib"
    INTERFACE_LINK_LIBRARIES "kh_ggml_base;kh_ggml_cpu;kh_ggml_opencl")

add_library(kh_llama_common_base STATIC IMPORTED GLOBAL)
set_target_properties(kh_llama_common_base PROPERTIES
    IMPORTED_LOCATION "${LLAMA_HW}/common/llama-common-base.lib")

add_library(kh_llama           STATIC IMPORTED GLOBAL)
set_target_properties(kh_llama PROPERTIES
    IMPORTED_LOCATION "${LLAMA_HW}/src/llama.lib"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/vendor/llama/include"
    INTERFACE_LINK_LIBRARIES "kh_ggml;kh_llama_common_base")

# Link-only smoke: if LNK2038 fires here, check /MT vs /MD between kh_* and KHANARY.
add_executable(KHANARY_LLM_SMOKE src/tests/test_llm_smoke.cpp)
target_link_libraries(KHANARY_LLM_SMOKE PRIVATE kh_llama)

# In-process inference wrapper: load a GGUF and run generate() without a server.
add_executable(KHANARY_LLM_INFER_TEST src/tests/test_llm_infer.cpp src/llm/llama_infer.cpp)
target_include_directories(KHANARY_LLM_INFER_TEST PRIVATE src)
target_link_libraries(KHANARY_LLM_INFER_TEST PRIVATE kh_llama)

