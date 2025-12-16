cmake_minimum_required(VERSION 3.22)
project(bootcamp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)

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
)

target_include_directories(bootcamp_core
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(bootcamp_core
        PUBLIC
        ${TORCH_LIBRARIES}
)
target_compile_definitions(bootcamp_core PRIVATE ENABLE_NVTX)


# === src main ===
add_executable(bootcamp_main
        src/main.cpp
)
target_compile_definitions(bootcamp_main PRIVATE ENABLE_NVTX)


target_link_libraries(bootcamp_main
        PRIVATE
        bootcamp_core
)

set_target_properties(bootcamp_main PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

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

target_link_libraries(bootcamp_tests
        PRIVATE
        bootcamp_core
        GTest::gtest_main
)

gtest_discover_tests(bootcamp_tests)

set_target_properties(bootcamp_tests PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
