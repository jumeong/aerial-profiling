# CMakeLists.txt for windowedChEstNoDftSOfdmKernel standalone profiler
#
# Build instructions (on Linux with CUDA installed):
#
#   cd cuPHY/profiling/ch_est
#   chmod +x build.sh
#   ./build.sh
#
# Or manually:
#   mkdir -p build && cd build
#   cmake .. -DCUPHY_ROOT=<path-to>/cuPHY \
#            -DCMAKE_CUDA_ARCHITECTURES=89   # Ada Lovelace (RTX 4070 Super)
#   make -j$(nproc)
#
# Run:
#   ./ch_est_profiler 64T1L
#   ./ch_est_profiler 64T2L
#   ./ch_est_profiler 16T4L
#
# Profile with Nsight Systems:
#   nsys profile --stats=true ./ch_est_profiler 64T1L
# Profile with Nsight Compute:
#   ncu --set full ./ch_est_profiler 64T1L
#
# Nsight profile:
#   nsys profile --stats=true ./ch_est_profiler 64T1L
#   ncu --set full ./ch_est_profiler 64T1L
#
# Note:
#   CUPHY_SRC_DIR  should point to the directory containing cuphy.h, cuphy_internal.h, etc.
#                  (i.e., cuPHY/src/cuphy)
#   NVLOG_DIR      should point to the nvlog include directory
#                  (i.e., cuPHY/nvlog/include)
#
#   The profiler uses the translation-unit trick:
#   ch_est_profiler.cu directly includes ch_est.cu from the source tree.
#   Therefore ch_est.cu MUST NOT be compiled separately in this project.

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(ch_est_profiler LANGUAGES CXX CUDA)

# ---------------------------------------------------------------
# User-settable paths (override via -D on command line)
# ---------------------------------------------------------------
set(CUPHY_ROOT "" CACHE PATH "Root of the cuPHY repo (parent of src/, nvlog/, etc.)")

if(NOT CUPHY_ROOT)
    # Auto-detect: this file is at cuPHY/profiling/ch_est/
    get_filename_component(CUPHY_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
    message(STATUS "Auto-detected CUPHY_ROOT = ${CUPHY_ROOT}")
endif()

set(CUPHY_SRC_DIR    "${CUPHY_ROOT}/src/cuphy")
set(NVLOG_DIR        "${CUPHY_ROOT}/nvlog/include")
set(CUPHY_UTIL_DIR   "${CUPHY_ROOT}/util")

# ---------------------------------------------------------------
# CUDA architecture: default = Ada Lovelace (RTX 4070 Super = sm_89)
# Override with: -DCMAKE_CUDA_ARCHITECTURES=89
# ---------------------------------------------------------------
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
    message(STATUS "Defaulting CMAKE_CUDA_ARCHITECTURES = 89 (Ada Lovelace)")
endif()

# ---------------------------------------------------------------
# C++ / CUDA standards
# ---------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------
# CUDA flags
# ---------------------------------------------------------------
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
    --extended-lambda \
    -Xcompiler -fPIC \
    -lineinfo \
    --generate-line-info \
    --ptxas-options=--warn-on-local-memory-usage \
    --ptxas-options=--warn-on-spills \
    -O3"
)

# ---------------------------------------------------------------
# External dependencies: cuRAND (bundled with CUDA Toolkit)
# ---------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)

# ---------------------------------------------------------------
# Executable
# Note: ch_est.cu is included via #include inside ch_est_profiler.cu
#       so it must NOT appear here as a separate source file.
# ---------------------------------------------------------------
add_executable(ch_est_profiler
    ch_est_profiler.cu
)

# ---------------------------------------------------------------
# Include paths
# ---------------------------------------------------------------
target_include_directories(ch_est_profiler PRIVATE
    # cuPHY core source (cuphy.h, cuphy_internal.h, cuphy_api.h, utils.cuh, etc.)
    "${CUPHY_SRC_DIR}"
    # ch_est-specific headers (ch_est.hpp, ch_est_types.hpp, ch_est_settings.hpp, etc.)
    "${CUPHY_SRC_DIR}/ch_est"
    # descrambling.cuh
    "${CUPHY_SRC_DIR}/descrambling"
    # type_convert.hpp
    "${CUPHY_SRC_DIR}"
    # nvlog.hpp, nvlog_fmt.hpp, memtrace.h
    "${NVLOG_DIR}"
    # gsl-lite (used in ch_est.hpp IModule.hpp)
    "${CUPHY_UTIL_DIR}/gsl-lite/include"
    # pusch_start_kernels.hpp
    "${CUPHY_SRC_DIR}"
)

# ---------------------------------------------------------------
# Link libraries
# ---------------------------------------------------------------
target_link_libraries(ch_est_profiler PRIVATE
    CUDA::cudart
    CUDA::curand
    CUDA::cuda_driver
)

# ---------------------------------------------------------------
# Compile definitions
# ---------------------------------------------------------------
target_compile_definitions(ch_est_profiler PRIVATE
    # Tell nvlog_fmt.hpp to suppress all logging in this standalone build.
    # NVLOGI_FMT / NVLOGC_FMT / NVLOGE_FMT will be no-ops.
    NVLOG_DISABLE_ALL
    # Suppress MemtraceDisableScope dependencies from linking against libmimalloc
    # (memtrace.h is header-only and uses weak symbols, so this is fine)
)

# ---------------------------------------------------------------
# Build info
# ---------------------------------------------------------------
message(STATUS "=== ch_est_profiler build configuration ===")
message(STATUS "  CUPHY_ROOT             : ${CUPHY_ROOT}")
message(STATUS "  CUPHY_SRC_DIR          : ${CUPHY_SRC_DIR}")
message(STATUS "  NVLOG_DIR              : ${NVLOG_DIR}")
message(STATUS "  CUPHY_UTIL_DIR         : ${CUPHY_UTIL_DIR}")
message(STATUS "  CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Build type             : ${CMAKE_BUILD_TYPE}")
message(STATUS "===========================================")
