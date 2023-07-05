cmake_minimum_required(VERSION 3.4.3)

set(SRC_FILES "ll_decomp.cu")
set(HEADER_FILES "BatchData.cuh")
project(ll_decomp  LANGUAGES C CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

find_package(CUDAToolkit REQUIRED)
find_package(nvcomp 2.6.1 REQUIRED)

add_executable("${PROJECT_NAME}" "${SRC_FILES}" "${HEADER_FILES}")
target_include_directories("${PROJECT_NAME}" PRIVATE
                           "${CMAKE_SOURCE_DIR}/src")
target_compile_options("${PROJECT_NAME}" PRIVATE -g -G)
target_link_libraries("${PROJECT_NAME}" PRIVATE nvcomp::nvcomp)

