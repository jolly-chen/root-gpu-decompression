cmake_minimum_required(VERSION 3.4.3)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

####################
# ROOT Compression #
####################

set(COMP_SRC_FILES "cpu_root_comp.cpp")
set(COMP_HEADER_FILES "")
project(cpu_root_comp  LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

find_package(ROOT REQUIRED COMPONENTS ROOTNTuple)

add_executable("${PROJECT_NAME}" "${COMP_SRC_FILES}" "${COMP_HEADER_FILES}")
target_include_directories("${PROJECT_NAME}" PRIVATE "${CMAKE_SOURCE_DIR}/src" "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_options("${PROJECT_NAME}" PRIVATE -g -lineinfo)
target_link_libraries("${PROJECT_NAME}" PRIVATE ROOT::ROOTNTuple)
