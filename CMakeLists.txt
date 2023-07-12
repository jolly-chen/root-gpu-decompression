cmake_minimum_required(VERSION 3.4.3)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

########################
# NVcomp Decompression #
########################

set(DECOMP_SRC_FILES "ll_decomp.cu")
set(DECOMP_HEADER_FILES "BatchData.cuh")
project(ll_decomp  LANGUAGES C CUDA CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

find_package(CUDAToolkit REQUIRED)
find_package(nvcomp 2.6.1 REQUIRED)

add_executable("${PROJECT_NAME}" "${DECOMP_SRC_FILES}" "${DECOMP_HEADER_FILES}")
target_include_directories("${PROJECT_NAME}" PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_link_libraries("${PROJECT_NAME}" PRIVATE nvcomp::nvcomp)

##############################################
# GPU decompression of ROOT compressed files #
##############################################

set(ROOT_DECOMP_SRC_FILES "gpu_root_decomp.cu")
set(ROOT_DECOMP_HEADER_FILES "")
project(gpu_root_decomp  LANGUAGES C CUDA CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

find_package(CUDAToolkit REQUIRED)
find_package(nvcomp 2.6.1 REQUIRED)
find_package(ROOT REQUIRED COMPONENTS Core)

add_executable("${PROJECT_NAME}" "${ROOT_DECOMP_SRC_FILES}" "${ROOT_DECOMP_HEADER_FILES}")
target_include_directories("${PROJECT_NAME}" PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_link_libraries("${PROJECT_NAME}" PRIVATE nvcomp::nvcomp ROOT::Core)

##############################################
# ROOT CPU decompression of ROOT compressed files #
##############################################

set(ROOT_DECOMP_SRC_FILES "cpu_root_decomp.cpp")
set(ROOT_DECOMP_HEADER_FILES "")
project(cpu_root_decomp  LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

find_package(ROOT REQUIRED COMPONENTS ROOTNTuple)

add_executable("${PROJECT_NAME}" "${ROOT_DECOMP_SRC_FILES}" "${ROOT_DECOMP_HEADER_FILES}")
target_include_directories("${PROJECT_NAME}" PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_link_libraries("${PROJECT_NAME}" PRIVATE ROOT::ROOTNTuple)

add_subdirectory(input)