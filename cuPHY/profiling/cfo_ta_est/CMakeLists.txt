cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(cfo_ta_est_profiler LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CUPHY_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
set(CUPHY_SRC_DIR "${CUPHY_ROOT}/src/cuphy")
set(CUPHY_UTIL_DIR "${CUPHY_ROOT}/util")
set(NVLOG_DIR "${CUPHY_ROOT}/src/nvlog")

# ---------------------------------------------------------------
# CUDA flags
# ---------------------------------------------------------------
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
    --extended-lambda \
    --expt-relaxed-constexpr \
    -Xcompiler -fPIC \
    -lineinfo \
    --generate-line-info \
    --ptxas-options=--warn-on-local-memory-usage \
    --ptxas-options=--warn-on-spills \
    -O3"
)

# ---------------------------------------------------------------
# External dependencies
# ---------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)

include(FetchContent)
FetchContent_Declare(
  gsl-lite
  GIT_REPOSITORY https://github.com/gsl-lite/gsl-lite.git
  GIT_TAG        v0.41.0
)
FetchContent_MakeAvailable(gsl-lite)

# ---------------------------------------------------------------
# Executable
# ---------------------------------------------------------------
add_executable(cfo_ta_est_profiler
    cfo_ta_est_profiler.cu
)

# ---------------------------------------------------------------
# Include paths
# ---------------------------------------------------------------
target_include_directories(cfo_ta_est_profiler PRIVATE
    "${CUPHY_SRC_DIR}"
    "${CUPHY_SRC_DIR}/cfo_ta_est"
    "${CUPHY_SRC_DIR}/ch_est" # In case cfo_ta_est uses some types from ch_est
    "${CUPHY_SRC_DIR}/descrambling"
    "${NVLOG_DIR}"
    "${CUPHY_UTIL_DIR}/aerial_utils"
    "${gsl-lite_SOURCE_DIR}/include"
    "${CUPHY_ROOT}/examples/common"
)

# ---------------------------------------------------------------
# Link libraries
# ---------------------------------------------------------------
target_link_libraries(cfo_ta_est_profiler PRIVATE
    CUDA::cudart
    CUDA::curand
    CUDA::cuda_driver
)

# ---------------------------------------------------------------
# Defines
# ---------------------------------------------------------------
target_compile_definitions(cfo_ta_est_profiler PRIVATE
    NVLOG_DISABLE_ALL
    gsl_CONFIG_DEFAULTS_VERSION=1
)
