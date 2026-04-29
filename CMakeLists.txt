cmake_minimum_required(VERSION 3.20)
project(cuda_kernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(vector_add vector_add.cu)
add_executable(matmul_naive matmul_naive.cu)
add_executable(matmul_tiled matmul_tiled.cu)
