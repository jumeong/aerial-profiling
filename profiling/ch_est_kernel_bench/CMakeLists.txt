cmake_minimum_required(VERSION 3.18)
project(bench_windowed_chest LANGUAGES CXX CUDA)

# ── CUDA settings ────────────────────────────────────────────────────────────
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD  17)

# Adjust -arch to match your GPU:
#   sm_80 = A100, sm_86 = A10/A30/RTX30xx, sm_89 = RTX4070 Super (Ada Lovelace), sm_90 = H100
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

# ── Target ───────────────────────────────────────────────────────────────────
add_executable(bench_windowed_chest bench_windowed_chest.cu)

target_compile_options(bench_windowed_chest PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        -lineinfo          # keeps line info for ncu source view
        --use_fast_math
        -Xcompiler -Wall
    >
)

# cooperative_groups requires linking against cudart
find_package(CUDAToolkit REQUIRED)
target_link_libraries(bench_windowed_chest PRIVATE CUDA::cudart)

# ── Install / run instructions (informational) ────────────────────────────────
message(STATUS "")
message(STATUS "Build:   cmake -B build && cmake --build build -j")
message(STATUS "Profile (Nsight Systems):  nsys profile --trace=cuda ./bench_windowed_chest")
message(STATUS "Profile (Nsight Compute):  ncu --set full -o chest_ncu ./bench_windowed_chest")
message(STATUS "")
