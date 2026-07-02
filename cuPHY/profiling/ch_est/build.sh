#!/usr/bin/env bash
# build.sh — Builds the ch_est_profiler
#
# Usage:
#   chmod +x build.sh
#   ./build.sh [clean]
#
# Environment variables:
#   CUDA_ARCH   : CUDA compute capability (default: 89 for RTX 4070 Super)
#   BUILD_TYPE  : Debug or Release (default: Release)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUPHY_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CUDA_ARCH="${CUDA_ARCH:-89}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
BUILD_DIR="${SCRIPT_DIR}/build"

echo "==========================="
echo " ch_est_profiler build"
echo "==========================="
echo " CUPHY_ROOT  : ${CUPHY_ROOT}"
echo " CUDA_ARCH   : sm_${CUDA_ARCH}"
echo " BUILD_TYPE  : ${BUILD_TYPE}"
echo " BUILD_DIR   : ${BUILD_DIR}"
echo "==========================="

if [ "${1}" == "clean" ]; then
    echo "[clean] Removing ${BUILD_DIR}"
    rm -rf "${BUILD_DIR}"
fi

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake "${SCRIPT_DIR}" \
    -DCUPHY_ROOT="${CUPHY_ROOT}" \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"

make -j"$(nproc)" ch_est_profiler

echo ""
echo "[Done] Binary: ${BUILD_DIR}/ch_est_profiler"
echo ""
echo "Run:"
echo "  ${BUILD_DIR}/ch_est_profiler 64T1L"
echo "  ${BUILD_DIR}/ch_est_profiler 64T2L"
echo "  ${BUILD_DIR}/ch_est_profiler 16T4L"
echo ""
echo "With Nsight Systems:"
echo "  nsys profile --stats=true ${BUILD_DIR}/ch_est_profiler 64T1L"
echo ""
echo "With Nsight Compute:"
echo "  ncu --set full ${BUILD_DIR}/ch_est_profiler 64T1L"
