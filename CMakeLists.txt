cmake_minimum_required(VERSION 3.4.3)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

########################
# NVcomp Decompression #
########################

set(DECOMP_SRC_FILES "ll_decomp.cu")
project(ll_decomp  LANGUAGES C CUDA CXX)

find_package(CUDAToolkit REQUIRED)
find_package(nvcomp 2.6.1 REQUIRED)

add_executable("${PROJECT_NAME}" "${DECOMP_SRC_FILES}")
target_link_libraries("${PROJECT_NAME}" PRIVATE nvcomp::nvcomp)

##############################################
# GPU decompression of ROOT compressed files #
##############################################

set(ROOT_DECOMP_SRC_FILES "gpu_root_decomp.cu")
project(gpu_root_decomp  LANGUAGES C CUDA CXX)

find_package(CUDAToolkit REQUIRED)
find_package(nvcomp 2.6.1 REQUIRED)
find_package(ROOT REQUIRED COMPONENTS Core)

add_executable("${PROJECT_NAME}" "${ROOT_DECOMP_SRC_FILES}")
target_link_libraries("${PROJECT_NAME}" PRIVATE nvcomp::nvcomp ROOT::Core)
target_compile_options("${PROJECT_NAME}" PRIVATE)

##############################################
# ROOT CPU decompression of ROOT compressed files #
##############################################

set(ROOT_DECOMP_SRC_FILES "cpu_root_decomp.cpp")
project(cpu_root_decomp  LANGUAGES C CXX)

find_package(ROOT REQUIRED COMPONENTS ROOTNTuple)

add_executable("${PROJECT_NAME}" "${ROOT_DECOMP_SRC_FILES}")
target_link_libraries("${PROJECT_NAME}" PRIVATE ROOT::ROOTNTuple)



##############################################
# ROOT CPU decompression of ROOT compressed files parallelised with openmp #
##############################################

set(ROOT_DECOMP2_SRC_FILES "cpu_root_decomp_omp.cpp")
project(cpu_root_decomp_omp  LANGUAGES C CXX)

find_package(ROOT REQUIRED COMPONENTS ROOTNTuple)
find_package(OpenMP REQUIRED)

add_executable("${PROJECT_NAME}" "${ROOT_DECOMP2_SRC_FILES}")
target_link_libraries("${PROJECT_NAME}" PRIVATE ROOT::ROOTNTuple OpenMP::OpenMP_CXX)

##############################################
# Tests for GPU packing
##############################################

project(pack_test LANGUAGES CUDA)

add_executable("${PROJECT_NAME}" pack_test.cu)

##############################################


add_subdirectory(input)