cmake_minimum_required(VERSION 3.18)
project(cuda-raytracer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Ensure we compile device code for Ampere (RTX 30-series) and also include a PTX for future
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86-real 86-virtual)

find_package(CUDAToolkit REQUIRED)

add_executable(cuda-raytracer
    main.cpp
    kernels.cu
)

# Include headers
target_include_directories(cuda-raytracer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Provide CUDA includes to host compiler
target_include_directories(cuda-raytracer PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

# Link CUDA runtime
target_link_libraries(cuda-raytracer PRIVATE CUDA::cudart)

# Position independent code just in case
set_target_properties(cuda-raytracer PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Prefer separate compilation
set_target_properties(cuda-raytracer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Warnings
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cuda-raytracer PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()
