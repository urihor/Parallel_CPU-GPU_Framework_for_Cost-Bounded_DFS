

cmake_minimum_required(VERSION 3.22)
project(bootcamp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(bootcamp_core
        src/puzzle_env.cpp
        src/puzzle15_state.cpp
        src/pdb15.cpp

)

target_include_directories(bootcamp_core
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)




enable_testing()


add_executable(bootcamp_tests
    tests/test_main.cpp
    tests/tests.cpp
    tests/test_generate_work.cpp
)

target_link_libraries(bootcamp_tests PRIVATE bootcamp_core)

# ����� �-ctest
add_test(NAME bootcamp_tests_run COMMAND bootcamp_tests)

# (���������) ������ ��� �������
set_target_properties(bootcamp_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
