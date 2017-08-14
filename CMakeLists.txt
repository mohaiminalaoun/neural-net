cmake_minimum_required(VERSION 3.7)
project(ML)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp Net.cpp Net.h)
add_executable(ML ${SOURCE_FILES})