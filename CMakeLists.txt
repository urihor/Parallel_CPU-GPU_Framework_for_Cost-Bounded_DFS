cmake_minimum_required(VERSION 3.22)
project(bootcamp LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Global options
# ---------------------------------------------------------------------------

# Control whether to build unit tests.
# On Windows we disable tests by default to avoid issues with FetchContent/SSL.
option(BOOTCAMP_BUILD_TESTS "Build unit tests" ON)
if (WIN32)
    set(BOOTCAMP_BUILD_TESTS OFF CACHE BOOL "Build unit tests" FORCE)
endif()

# Use C++20 across the project.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optional: emit compile_commands.json (helps IDEs and tooling).
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# Feature flags / options
# ---------------------------------------------------------------------------

# Enable NVTX annotations (for Nsight Systems/Compute).
# This requires nvToolsExt headers from the CUDA Toolkit.
option(BOOTCAMP_ENABLE_NVTX "Enable NVTX annotations (requires CUDA Toolkit nvToolsExt)" ON)

# ---------------------------------------------------------------------------
# Torch (LibTorch)
# ---------------------------------------------------------------------------

# Find the LibTorch installation (path is typically provided via Torch_DIR).
find_package(Torch REQUIRED)

# Torch may require additional compiler flags; append them to the global flags.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# ---------------------------------------------------------------------------
# Helper: consistent output directories (libs/bin) for all targets
# ---------------------------------------------------------------------------

function(bootcamp_set_output_dirs target)
    # Default output dirs for single-config generators (e.g. Ninja, Makefiles).
    set_target_properties(${target} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
            ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    )
    # For multi-config generators (e.g. Visual Studio), set per-configuration dirs.
    if (CMAKE_CONFIGURATION_TYPES)
        foreach(conf ${CMAKE_CONFIGURATION_TYPES})
            string(TOUPPER "${conf}" conf_uc)
            set_target_properties(${target} PROPERTIES
                    "RUNTIME_OUTPUT_DIRECTORY_${conf_uc}" "${CMAKE_BINARY_DIR}/bin/${conf}"
                    "LIBRARY_OUTPUT_DIRECTORY_${conf_uc}" "${CMAKE_BINARY_DIR}/bin/${conf}"
                    "ARCHIVE_OUTPUT_DIRECTORY_${conf_uc}" "${CMAKE_BINARY_DIR}/lib/${conf}"
            )
        endforeach()
    endif()
endfunction()

# ---------------------------------------------------------------------------
# Core library (bootcamp_core)
#
# This is the main C++ library that holds:
#   - 15-puzzle domain (state, environment, Manhattan)
#   - PDB build + lookup code
#   - Neural heuristics (TorchScript / LibTorch bindings)
#   - App logic
# ---------------------------------------------------------------------------

add_library(bootcamp_core
        src/puzzle_env.cpp
        src/puzzle15_state.cpp
        src/pdb15.cpp
        src/do_iteration.cpp
        src/solution_printer.cpp
        src/korf_examples.cpp
        src/neural_delta_15_quantile.cpp
        src/neural_batch_service.cpp
        src/manhattan_15.cpp
        src/deepcubea15_heuristic.cpp
        src/app.cpp
        src/nn_over_corrections.cpp
)

target_include_directories(bootcamp_core
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(bootcamp_core
        PUBLIC
        ${TORCH_LIBRARIES}
)

# Windows-specific definitions to keep Windows headers smaller and avoid
# min/max macro collisions with std::min/std::max.
if (WIN32)
    target_compile_definitions(bootcamp_core PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# ---------------------------------------------------------------------------
# NVTX (NVTX v3 header-only integration)
#
# If BOOTCAMP_ENABLE_NVTX is ON and the nvtx3 headers are found, we define
# ENABLE_NVTX and add the include path so that NVTX markers are compiled in.
# Otherwise NVTX is silently disabled.
# ---------------------------------------------------------------------------

set(BOOTCAMP_USE_NVTX ${BOOTCAMP_ENABLE_NVTX})

set(BOOTCAMP_NVTX3_INCLUDE_DIR "")
if (BOOTCAMP_USE_NVTX)
    find_path(BOOTCAMP_NVTX3_INCLUDE_DIR
            NAMES nvtx3/nvToolsExt.h
            HINTS
            "${CUDAToolkit_ROOT}/include"
            "${CUDA_TOOLKIT_ROOT_DIR}/include"
            "$ENV{CUDA_PATH}/include"
            "/usr/local/cuda/include"
    )

    if (NOT BOOTCAMP_NVTX3_INCLUDE_DIR)
        message(WARNING "NVTX enabled but nvtx3/nvToolsExt.h not found -> NVTX disabled.")
        set(BOOTCAMP_USE_NVTX OFF)
    endif()
endif()

if (BOOTCAMP_USE_NVTX)
    target_compile_definitions(bootcamp_core PRIVATE ENABLE_NVTX)
    target_include_directories(bootcamp_core PRIVATE ${BOOTCAMP_NVTX3_INCLUDE_DIR})
    # On Linux we also need libdl for nvToolsExt dynamic loading.
    if (UNIX AND NOT APPLE)
        target_link_libraries(bootcamp_core PRIVATE dl)
    endif()
endif()

# ---------------------------------------------------------------------------
# Main executable (bootcamp_main)
#
# Thin CLI wrapper that:
#   * parses command-line arguments
#   * calls into bootcamp_core::run(...)
# ---------------------------------------------------------------------------

add_executable(bootcamp_main
        src/main.cpp
)

target_link_libraries(bootcamp_main
        PRIVATE
        bootcamp_core
)

if (BOOTCAMP_USE_NVTX)
    target_compile_definitions(bootcamp_main PRIVATE ENABLE_NVTX)
endif()

bootcamp_set_output_dirs(bootcamp_main)

if (BOOTCAMP_USE_NVTX)
    target_compile_definitions(bootcamp_main PRIVATE ENABLE_NVTX)
    target_include_directories(bootcamp_main PRIVATE ${BOOTCAMP_NVTX3_INCLUDE_DIR})
    if (UNIX AND NOT APPLE)
        target_link_libraries(bootcamp_main PRIVATE dl)
    endif()
endif()

# ---------------------------------------------------------------------------
# GoogleTest-based unit tests (bootcamp_tests)
#
# Uses FetchContent to download GoogleTest at configure time.
# The tests link against bootcamp_core and can exercise the full solver stack.
# ---------------------------------------------------------------------------

if (BOOTCAMP_BUILD_TESTS)
    include(FetchContent)

    FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )

    # Use the shared CRT on Windows for GoogleTest.
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()
    include(GoogleTest)

    add_executable(bootcamp_tests
            tests/test_main.cpp
            tests/tests.cpp
            tests/test_generate_work.cpp
            tests/test_do_iteration.cpp
            tests/test_cb-dfs.cpp
            tests/test_ida_star_korf_boards.cpp
    )

    if (BOOTCAMP_USE_NVTX)
        target_compile_definitions(bootcamp_tests PRIVATE ENABLE_NVTX)
        target_include_directories(bootcamp_tests PRIVATE ${BOOTCAMP_NVTX3_INCLUDE_DIR})
        if (UNIX AND NOT APPLE)
            target_link_libraries(bootcamp_tests PRIVATE dl)
        endif()
    endif()

    target_link_libraries(bootcamp_tests
            PRIVATE
            bootcamp_core
            GTest::gtest_main
    )

    # Automatically discover and register GTest tests.
    gtest_discover_tests(bootcamp_tests)

    set_target_properties(bootcamp_tests PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endif()

# ---------------------------------------------------------------------------
# Windows: copy LibTorch DLLs next to the executables after build
#
# This ensures that running bootcamp_main / bootcamp_tests directly from the
# build tree finds all the necessary Torch DLLs at runtime.
# ---------------------------------------------------------------------------

if (WIN32)
    # TorchConfig usually defines TORCH_INSTALL_PREFIX.
    if (DEFINED TORCH_INSTALL_PREFIX)
        file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    else()
        # Fallback for some distributions/layouts:
        file(GLOB TORCH_DLLS "${TORCH_DIR}/../../../lib/*.dll")
    endif()

    if (TORCH_DLLS)
        add_custom_command(TARGET bootcamp_main POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${TORCH_DLLS}
                $<TARGET_FILE_DIR:bootcamp_main>
                VERBATIM
        )

        if (BOOTCAMP_BUILD_TESTS)
            add_custom_command(TARGET bootcamp_tests POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${TORCH_DLLS}
                    $<TARGET_FILE_DIR:bootcamp_tests>
                    VERBATIM
            )
        endif()
    endif()

endif()

# ---------------------------------------------------------------------------
# build_nn_over_corrections tool
#
# Small helper executable that builds correction tables for neural heuristics.
# It links directly against LibTorch (no dependency on bootcamp_core headers)
# so it can be used as a standalone offline preprocessing step.
# ---------------------------------------------------------------------------

add_executable(build_nn_over_corrections
        src/build_nn_over_corrections.cpp
)
target_link_libraries(build_nn_over_corrections PRIVATE ${TORCH_LIBRARIES})
target_include_directories(build_nn_over_corrections PRIVATE ${TORCH_INCLUDE_DIRS})
set_property(TARGET build_nn_over_corrections PROPERTY CXX_STANDARD 20)
