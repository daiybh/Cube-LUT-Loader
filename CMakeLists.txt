cmake_minimum_required(VERSION 3.23)

option(BUILD_TESTS "Build tests" OFF)
option(BUILD_CUDA "Build project with CUDA libraries" OFF)

if(BUILD_CUDA)
    project(lut_loader VERSION 0.1.0 LANGUAGES CUDA CXX)
    add_compile_definitions(BUILD_CUDA)
else()
    project(lut_loader VERSION 0.1.0 LANGUAGES CXX)
endif()

if(BUILD_TESTS)
        enable_testing()
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set(SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)
set(MAIN_LIB_NAME lut_loader_base)
set(LIBRARIES_TO_LINK opencv_core opencv_imgcodecs CLI11::CLI11 Eigen3::Eigen)
add_subdirectory(src)
