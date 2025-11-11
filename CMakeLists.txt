cmake_minimum_required(VERSION 3.22)
project(bootcamp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(bootcamp
        main.cpp
        tests.cpp
        puzzle_env.cpp
        puzzle15_state.cpp
        puzzle_env.h
        puzzle15_state.h
        puzzle_actions.h
        i_env.h
        work.h
        generate_work.h
)