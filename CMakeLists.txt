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
    src/kuhul/kuhul_vm.cpp
    src/kuhul/xcfe_executor.cpp
    src/kuhul/gram_store.cpp
    src/klsl/klsl_lexer.cpp
    src/klsl/klsl_parser.cpp
    src/klsl/klsl_ir.cpp
    src/klsl/hlsl_codegen.cpp
    src/klsl/wgsl_codegen.cpp
    src/klsl/d3dcompile_bridge.cpp
    src/gpu/d3d11_device.cpp
    src/gpu/dml_ops.cpp
    src/gpu/tensor_buffer.cpp
    src/gpu/compute_dispatch.cpp
    src/gpu/dds_loader.cpp
    src/gpu/hot_swap.cpp
    src/inference/tokenizer.cpp
    src/inference/moe_router.cpp
    src/inference/inference_engine.cpp
    src/inference/kv_cache.cpp
    src/inference/sampler.cpp
    src/micronaut/micronaut_registry.cpp
    src/micronaut/aiml_engine.cpp
    src/micronaut/eliza_operators.cpp
    src/micronaut/track_loader.cpp
    src/micronaut/sandbox_inspector.cpp
    src/micronaut/gravity_field.cpp
    src/world/world_state.cpp
    src/world/field_dag.cpp
    src/micronaut/factory.cpp
    src/memory/perception.cpp
    src/memory/confidence.cpp
    src/memory/semantic_graph.cpp
    src/memory/scoring.cpp
    src/io/json_rpc_server.cpp
    src/io/mcp_server.cpp
    src/io/jrom.cpp
    src/io/request_router.cpp
    src/util/sha256.cpp
    src/util/json.cpp
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
