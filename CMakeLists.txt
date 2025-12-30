cmake_minimum_required(VERSION 3.22)
project(bootcamp LANGUAGES CXX)
# Build tests (default ON on Linux, OFF on Windows to avoid FetchContent/SSL issues)
option(BOOTCAMP_BUILD_TESTS "Build unit tests" ON)
if (WIN32)
    set(BOOTCAMP_BUILD_TESTS OFF CACHE BOOL "Build unit tests" FORCE)
endif()


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optional: makes it easier for tools/IDEs.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# -------------------------
# Options
# -------------------------
option(BOOTCAMP_ENABLE_NVTX "Enable NVTX annotations (requires CUDA Toolkit nvToolsExt)" ON)

# -------------------------
# Torch
# -------------------------
find_package(Torch REQUIRED)
# Torch may provide additional required compiler flags.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# -------------------------
# Helper: output dirs (supports multi-config generators like Visual Studio)
# -------------------------
function(bootcamp_set_output_dirs target)
    set_target_properties(${target} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
            ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    )
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

# -------------------------
# Core library
# -------------------------
add_library(bootcamp_core
        src/puzzle_env.cpp
        src/puzzle15_state.cpp
        src/pdb15.cpp
        src/do_iteration.cpp
        src/solution_printer.cpp
        src/korf_examples.cpp
        src/neural_delta_15.cpp
        src/neural_batch_service.cpp
        src/manhattan_15.cpp
        src/neural_delta_15_quantile.cpp
        src/nn_over_corrections.cpp
)
if (BOOTCAMP_ENABLE_NVTX)
    target_compile_definitions(bootcamp_core PUBLIC ENABLE_NVTX)
    target_include_directories(bootcamp_core PUBLIC "D:/include")
endif()


target_include_directories(bootcamp_core
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(bootcamp_core
        PUBLIC
        ${TORCH_LIBRARIES}
)
if (WIN32)
    target_compile_definitions(bootcamp_core PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# -------------------------
# NVTX (NVTX v3 header-only)
# -------------------------
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
    if (UNIX AND NOT APPLE)
        target_link_libraries(bootcamp_core PRIVATE dl)
    endif()
endif()


# -------------------------
# Main executable
# -------------------------
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


# -------------------------
# GoogleTest
# -------------------------
if (BOOTCAMP_BUILD_TESTS)
    include(FetchContent)

    FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )

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

    gtest_discover_tests(bootcamp_tests)

    set_target_properties(bootcamp_tests PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endif()


# -------------------------
# Windows: copy LibTorch DLLs next to the executables after build
# -------------------------
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

add_executable(build_nn_over_corrections
        src/build_nn_over_corrections.cpp
)
target_link_libraries(build_nn_over_corrections PRIVATE ${TORCH_LIBRARIES})
target_include_directories(build_nn_over_corrections PRIVATE ${TORCH_INCLUDE_DIRS})
set_property(TARGET build_nn_over_corrections PROPERTY CXX_STANDARD 20)