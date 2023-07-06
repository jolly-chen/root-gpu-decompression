cmake_minimum_required(VERSION 3.4.3)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

#################
# Decompression #
#################

set(DECOMP_SRC_FILES "ll_decomp.cu")
set(DECOMP_HEADER_FILES "BatchData.cuh")
project(ll_decomp  LANGUAGES C CUDA CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

find_package(CUDAToolkit REQUIRED)
find_package(nvcomp 2.6.1 REQUIRED)

add_executable("${PROJECT_NAME}" "${DECOMP_SRC_FILES}" "${DECOMP_HEADER_FILES}")
target_include_directories("${PROJECT_NAME}" PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_compile_options("${PROJECT_NAME}" PRIVATE -g -G)
target_link_libraries("${PROJECT_NAME}" PRIVATE nvcomp::nvcomp)

#################
# ROOT Decompression #
#################

set(ROOT_DECOMP_SRC_FILES "root_decomp.cu")
set(ROOT_DECOMP_HEADER_FILES "BatchData.cuh")
project(root_decomp  LANGUAGES C CUDA CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

find_package(CUDAToolkit REQUIRED)
find_package(nvcomp 2.6.1 REQUIRED)

add_executable("${PROJECT_NAME}" "${ROOT_DECOMP_SRC_FILES}" "${ROOT_DECOMP_HEADER_FILES}")
target_include_directories("${PROJECT_NAME}" PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_compile_options("${PROJECT_NAME}" PRIVATE -g -G)
target_link_libraries("${PROJECT_NAME}" PRIVATE nvcomp::nvcomp)


###############
# Compression #
###############

set(COMP_SRC_FILES "root_comp.cpp")
set(COMP_HEADER_FILES "")
project(root_comp  LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

find_package(ROOT REQUIRED COMPONENTS ROOTNTuple)

add_executable("${PROJECT_NAME}" "${COMP_SRC_FILES}" "${COMP_HEADER_FILES}")
target_include_directories("${PROJECT_NAME}" PRIVATE "${CMAKE_SOURCE_DIR}/src" "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_options("${PROJECT_NAME}" PRIVATE -g -lineinfo)
target_link_libraries("${PROJECT_NAME}" PRIVATE ROOT::ROOTNTuple)
