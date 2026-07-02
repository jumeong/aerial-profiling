/*
 * ch_est_profiler.cu
 *
 * Standalone profiling harness for windowedChEstNoDftSOfdmKernel.
 *
 * Usage:
 *   ./ch_est_profiler 64T1L    -> 64 RX antennas, 1 layer
 *   ./ch_est_profiler 64T2L    -> 64 RX antennas, 2 layers
 *   ./ch_est_profiler 16T4L    -> 16 RX antennas, 4 layers
 *
 * Fixed config:
 *   - 273 RB, single-symbol DMRS (N_DMRS_SYMS=1)
 *   - DMRS symbol indices: 2 and 11 (launched separately)
 *   - N_DMRS_INTERP_PRB_OUT_PER_CLUSTER = 4  -> N_DMRS_PRB_IN_PER_CLUSTER = 8
 *   - N_DMRS_GRIDS_PER_PRB = 2 (DMRS Type 1)
 *   - 1 UE group
 *   - TStorage=float, TDataRx=__half, TCompute=float
 *
 * GPU scaling note:
 *   RTX 4070 Super (Ada Lovelace): 7168 CUDA cores, 56 SMs
 *   NVIDIA Spark GB10 (Blackwell): 6144 CUDA cores
 *   Scale factor: GB10 / RTX4070S ≈ 6144/7168 ≈ 0.86x (GB10 slightly slower, compute-bound estimate)
 */

// cuda_fp16.h must come before cuphy.hpp (via ch_est.cu) because cuphy.hpp
// uses __half22float2() which is declared in cuda_fp16.h.
#include "cuda_fp16.h"

// ============================================================
// Translation-unit trick: include ch_est.cu directly so that
// the static __global__ kernel symbols are visible in this TU.
// Do NOT add ch_est.cu as a separate compilation target.
// ============================================================
#include "ch_est.cu"

// ============================================================
// Additional system headers needed by this file
// ============================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cmath>
#include <string>
#include <curand.h>

// ============================================================
// Constants
// ============================================================
static constexpr uint32_t N_PRB              = 273;
static constexpr uint32_t N_TONES_PER_PRB_P = 12;
static constexpr uint32_t N_OFDM_SYMS       = 14;
static constexpr uint32_t N_UE_GRPS         = 1;
static constexpr uint32_t N_DMRS_GRIDS      = 2;    // DMRS Type 1
static constexpr uint32_t N_DMRS_SYMS_VAL   = 1;    // single-symbol DMRS
static constexpr uint32_t N_DMRS_IN         = 8;    // N_DMRS_PRB_IN_PER_CLUSTER
static constexpr uint32_t N_DMRS_OUT        = 4;    // N_DMRS_INTERP_PRB_OUT_PER_CLUSTER
static constexpr uint32_t WARMUP_ITERS      = 10;
static constexpr uint32_t TIMING_ITERS      = 100;
static constexpr uint32_t SLOT_NUM          = 0;

// GPU comparison (CUDA cores)
static constexpr double RTX4070S_CUDA_CORES = 7168.0;
static constexpr double GB10_CUDA_CORES     = 6144.0;  // NVIDIA Spark GB10 (Blackwell)

// ============================================================
// Error checking helpers
// ============================================================
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t _err = (call);                                               \
        if (_err != cudaSuccess) {                                               \
            fprintf(stderr, "[CUDA ERROR] %s:%d  %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(_err));               \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while(0)

#define CURAND_CHECK(call)                                                       \
    do {                                                                         \
        curandStatus_t _st = (call);                                             \
        if (_st != CURAND_STATUS_SUCCESS) {                                      \
            fprintf(stderr, "[CURAND ERROR] %s:%d  status=%d\n",                \
                    __FILE__, __LINE__, (int)_st);                               \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while(0)

// ============================================================
// Simple GPU memory helper
// ============================================================
template<typename T>
T* gpu_alloc(size_t n) {
    T* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
    CUDA_CHECK(cudaMemset(ptr, 0, n * sizeof(T)));
    return ptr;
}

template<typename T>
T* cpu_alloc(size_t n) {
    T* ptr = new T[n]();
    return ptr;
}

// ============================================================
// Fill GPU buffer with random FP32, then optionally cast to FP16
// ============================================================
static void fill_random_fp32(float* d_buf, size_t n, curandGenerator_t gen) {
    CURAND_CHECK(curandGenerateUniform(gen, d_buf, n));
}

// Simple GPU kernel to cast FP32 → FP16 (complex interleaved)
__global__ void castFp32ToFp16Kernel(const float* __restrict__ src,
                                      __half* __restrict__ dst,
                                      size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __float2half(src[idx]);
}

static void fill_random_fp16(__half* d_buf, size_t n, curandGenerator_t gen,
                              float* d_tmp) {
    // Use even n for simplicity
    fill_random_fp32(d_tmp, n, gen);
    dim3 blk(256);
    dim3 grd((n + 255) / 256);
    castFp32ToFp16Kernel<<<grd, blk>>>(d_tmp, d_buf, n);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================
// Set up tensor strides for a 3-D column-major tensor (d0, d1, d2)
// stride[0] = 1, stride[1] = d0, stride[2] = d0*d1
// ============================================================
static void set_strides3(int* strides, int d0, int d1) {
    strides[0] = 1;
    strides[1] = d0;
    strides[2] = d0 * d1;
}

// 4-D column-major: stride[k] = prod(d[0..k-1])
static void set_strides4(int* strides, int d0, int d1, int d2) {
    strides[0] = 1;
    strides[1] = d0;
    strides[2] = d0 * d1;
    strides[3] = d0 * d1 * d2;
}

// ============================================================
// Parse CLI argument "64T1L" -> nRxAnt=64, nLayers=1
// ============================================================
static bool parse_config(const char* arg, int& nRxAnt, int& nLayers) {
    // Expected format: <nRxAnt>T<nLayers>L
    char* end;
    long ant = strtol(arg, &end, 10);
    if (*end != 'T') return false;
    long lay = strtol(end + 1, &end, 10);
    if (*end != 'L') return false;
    nRxAnt  = (int)ant;
    nLayers = (int)lay;
    return true;
}

// ============================================================
// Template dispatch: launch kernel for a given N_LAYERS
// ============================================================
template<uint32_t N_LAYERS>
static void launch_kernel(ch_est::puschRxChEstStatDescr_t* d_stat,
                           ch_est::puschRxChEstDynDescr_t*  d_dyn,
                           dim3 gridDim, dim3 blockDim,
                           cudaStream_t stream) {
    ch_est::windowedChEstNoDftSOfdmKernel<
        float,      // TStorage
        __half,     // TDataRx
        float,      // TCompute
        N_LAYERS,   // N_LAYERS
        2,          // N_DMRS_GRIDS_PER_PRB  (Type 1)
        N_DMRS_IN,  // N_DMRS_PRB_IN_PER_CLUSTER
        N_DMRS_OUT, // N_DMRS_INTERP_PRB_OUT_PER_CLUSTER
        N_DMRS_SYMS_VAL // N_DMRS_SYMS
    ><<<gridDim, blockDim, 0, stream>>>(d_stat, d_dyn);
}

// ============================================================
// main()
// ============================================================
int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <config>\n", argv[0]);
        fprintf(stderr, "  config: 64T1L | 64T2L | 16T4L\n");
        return EXIT_FAILURE;
    }

    int nRxAnt = 0, nLayers = 0;
    if (!parse_config(argv[1], nRxAnt, nLayers)) {
        fprintf(stderr, "ERROR: Cannot parse config '%s'\n", argv[1]);
        fprintf(stderr, "  Expected format: <nRxAnt>T<nLayers>L (e.g. 64T1L)\n");
        return EXIT_FAILURE;
    }

    printf("=======================================================\n");
    printf(" windowedChEstNoDftSOfdmKernel Profiler\n");
    printf("=======================================================\n");
    printf(" Config   : %s\n", argv[1]);
    printf(" nRxAnt   : %d\n", nRxAnt);
    printf(" nLayers  : %d\n", nLayers);
    printf(" nPRB     : %d\n", N_PRB);
    printf(" DMRS syms: [2, 11]\n");
    printf(" Warmup   : %d iters\n", WARMUP_ITERS);
    printf(" Timing   : %d iters\n", TIMING_ITERS);
    printf("=======================================================\n\n");

    // ----------------------------------------------------------
    // GPU query
    // ----------------------------------------------------------
    int deviceId = 0;
    CUDA_CHECK(cudaGetDevice(&deviceId));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));
    printf("[GPU] %s (SM %d.%d, %d SMs, %.0f MHz boost)\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount,
           prop.clockRate / 1000.0f);
    printf("[GPU] CUDA cores = SMs x 128 = %d (approx)\n\n",
           prop.multiProcessorCount * 128);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // ----------------------------------------------------------
    // cuRAND generator for synthetic data
    // ----------------------------------------------------------
    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, 12345ULL));

    // ----------------------------------------------------------
    // Tensor size calculations
    // ----------------------------------------------------------
    // tDataRx: (NF, ND, nRxAnt) complex FP16
    //   NF = N_PRB_CLUSTERS * N_DMRS_OUT * N_TONES_PER_PRB = 69 * 4 * 12 = 3312
    //   ND = N_OFDM_SYMS = 14
    //   NOTE: We allocate up to the rounded-up PRB cluster size because the kernel 
    //         writes up to N_PRB_CLUSTERS * N_DMRS_OUT PRBs to tHEst, causing out-of-bounds
    //         if we only allocate N_PRB (273).
    const int N_PRB_CLUSTERS = (N_PRB + N_DMRS_OUT - 1) / N_DMRS_OUT;
    const int NF = N_PRB_CLUSTERS * N_DMRS_OUT * N_TONES_PER_PRB_P; // 3312
    const int ND = N_OFDM_SYMS;               // 14
    const size_t n_dataRx = (size_t)NF * ND * nRxAnt;

    // tFreqInterpCoefs: (49, 48, 3) float  [for N_DMRS_PRB_IN=8]
    //   dim0 = N_TOTAL_DMRS_INTERP_GRID_TONES + N_INTER_DMRS_GRID_FREQ_SHIFT
    //        = N_DMRS_OUT * N_TONES_PER_PRB * N_DMRS_GRIDS + 1 = 4*12*2 + 1 = 97... wait
    // Let me recalculate:
    //   N_DMRS_INTERP_TONES_PER_GRID = N_TONES_PER_PRB = 12
    //   N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER = N_DMRS_INTERP_TONES_PER_GRID * N_DMRS_OUT = 12*4 = 48
    //   N_INTER_DMRS_GRID_FREQ_SHIFT = 1 (for N_DMRS_GRIDS_PER_PRB=2)
    //   dim0 = 48 + 1 = 49
    //   N_DMRS_GRID_TONES_PER_PRB = N_TONES_PER_PRB / N_DMRS_GRIDS = 12/2 = 6
    //   N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER = N_DMRS_GRID_TONES_PER_PRB * N_DMRS_IN = 6*8 = 48
    //   dim1 = 48
    //   dim2 = 3 (filters: middle, lower-edge, upper-edge)
    const int COEF_D0 = 49; // N_TOTAL_DMRS_INTERP_GRID_TONES + N_INTER_DMRS_GRID_FREQ_SHIFT
    const int COEF_D1 = 48; // N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER
    const int COEF_D2 = 3;
    const size_t n_freqInterp = (size_t)COEF_D0 * COEF_D1 * COEF_D2;

    // tShiftSeq: (N_PRB * N_DMRS_GRID_TONES_PER_PRB, N_DMRS_SYMS) complex FP16
    //   = (273*6, 1) = (1638, 1)
    const int SHIFT_D0 = N_PRB * (N_TONES_PER_PRB_P / N_DMRS_GRIDS); // 273*6 = 1638
    const int SHIFT_D1 = N_DMRS_SYMS_VAL;                             // 1
    const size_t n_shiftSeq = (size_t)SHIFT_D0 * SHIFT_D1;           // complex, so *2 for FP16

    // tUnShiftSeq: (N_PRB * N_TONES_PER_PRB * N_DMRS_GRIDS + N_INTER_DMRS_GRID_FREQ_SHIFT,) complex FP16
    //   = (273*12*2 + 1,) = (6553,)
    const int UNSHIFT_D0 = N_PRB * N_TONES_PER_PRB_P * N_DMRS_GRIDS + 1; // 6553
    const size_t n_unShiftSeq = (size_t)UNSHIFT_D0;

    // tHEst: (nRxAnt, nLayers, N_PRB*12, 1) complex FP32
    //   (stored as N_SC=NF column-major)
    const size_t n_hEst = (size_t)nRxAnt * nLayers * NF * 1; // NH=1

    // ----------------------------------------------------------
    // Allocate GPU tensors
    // ----------------------------------------------------------
    printf("[ALLOC] Allocating GPU tensors...\n");

    // Allocate FP32 temporary buffer for cuRAND (reused across allocations)
    float* d_tmp;
    {
        size_t n_tmp = std::max({n_dataRx * 2, n_freqInterp,
                                  n_shiftSeq * 2, n_unShiftSeq * 2, n_hEst * 2});
        CUDA_CHECK(cudaMalloc(&d_tmp, n_tmp * sizeof(float)));
    }

    // tDataRx (FP16 complex)
    __half* d_dataRx = gpu_alloc<__half>(n_dataRx * 2); // *2 for complex (re+im interleaved)
    fill_random_fp16(d_dataRx, n_dataRx * 2, gen, d_tmp);

    // tFreqInterpCoefs (FP32 scalar, used as real-valued filter coefficients)
    float* d_freqInterp = gpu_alloc<float>(n_freqInterp);
    fill_random_fp32(d_freqInterp, n_freqInterp, gen);

    // tShiftSeq (FP16 complex)
    __half* d_shiftSeq = gpu_alloc<__half>(n_shiftSeq * 2);
    fill_random_fp16(d_shiftSeq, n_shiftSeq * 2, gen, d_tmp);

    // tUnShiftSeq (FP16 complex)
    __half* d_unShiftSeq = gpu_alloc<__half>(n_unShiftSeq * 2);
    fill_random_fp16(d_unShiftSeq, n_unShiftSeq * 2, gen, d_tmp);

    // tHEst (FP32 complex = cuComplex)
    cuComplex* d_hEst = gpu_alloc<cuComplex>(n_hEst);

    // tChEstDbg (FP32 complex, dummy — same shape as tHEst)
    cuComplex* d_chEstDbg = gpu_alloc<cuComplex>(n_hEst);

    printf("[ALLOC] Done.\n\n");

    // ----------------------------------------------------------
    // Stride arrays (must be on GPU for tensor_ref in device code)
    // We allocate them in pinned memory and copy to device.
    // ----------------------------------------------------------
    // All stride arrays: max 4 dims
    int strides_dataRx[3], strides_freqInterp[3], strides_shiftSeq[2],
        strides_unShiftSeq[1], strides_hEst[4], strides_dbg[4];

    // tDataRx: (NF, ND, nRxAnt) complex FP16 — strides in elements
    // Since complex is stored as two consecutive __half values:
    // stride[0] = 1 complex = 1 element index step
    set_strides3(strides_dataRx, NF, ND);

    // tFreqInterpCoefs: (COEF_D0, COEF_D1, COEF_D2) float scalar
    set_strides3(strides_freqInterp, COEF_D0, COEF_D1);

    // tShiftSeq: (SHIFT_D0, SHIFT_D1) complex FP16
    strides_shiftSeq[0] = 1;
    strides_shiftSeq[1] = SHIFT_D0;

    // tUnShiftSeq: (UNSHIFT_D0,) complex FP16
    strides_unShiftSeq[0] = 1;

    // tHEst: (nRxAnt, nLayers, NF, 1) cuComplex
    set_strides4(strides_hEst, nRxAnt, nLayers, NF);

    // tDbg: same shape
    set_strides4(strides_dbg, nRxAnt, nLayers, NF);

    // Copy stride arrays to device
    int* d_strides_dataRx;   CUDA_CHECK(cudaMalloc(&d_strides_dataRx,   3 * sizeof(int)));
    int* d_strides_freqInterp;CUDA_CHECK(cudaMalloc(&d_strides_freqInterp, 3 * sizeof(int)));
    int* d_strides_shiftSeq; CUDA_CHECK(cudaMalloc(&d_strides_shiftSeq, 2 * sizeof(int)));
    int* d_strides_unShift;  CUDA_CHECK(cudaMalloc(&d_strides_unShift,  1 * sizeof(int)));
    int* d_strides_hEst;     CUDA_CHECK(cudaMalloc(&d_strides_hEst,     4 * sizeof(int)));
    int* d_strides_dbg;      CUDA_CHECK(cudaMalloc(&d_strides_dbg,      4 * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_strides_dataRx,    strides_dataRx,    3*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_strides_freqInterp, strides_freqInterp, 3*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_strides_shiftSeq,  strides_shiftSeq,  2*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_strides_unShift,   strides_unShiftSeq,1*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_strides_hEst,      strides_hEst,      4*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_strides_dbg,       strides_dbg,       4*sizeof(int), cudaMemcpyHostToDevice));

    // ----------------------------------------------------------
    // Build Static Descriptor (CPU → GPU)
    // ----------------------------------------------------------
    ch_est::puschRxChEstStatDescr_t h_statDescr;
    memset(&h_statDescr, 0, sizeof(h_statDescr));

    // tPrmFreqInterpCoefs (N_DMRS_PRB_IN=8 path uses tPrmFreqInterpCoefs)
    h_statDescr.tPrmFreqInterpCoefs.pAddr      = d_freqInterp;
    h_statDescr.tPrmFreqInterpCoefs.strides[0] = strides_freqInterp[0];
    h_statDescr.tPrmFreqInterpCoefs.strides[1] = strides_freqInterp[1];
    h_statDescr.tPrmFreqInterpCoefs.strides[2] = strides_freqInterp[2];

    // tPrmShiftSeq
    h_statDescr.tPrmShiftSeq.pAddr      = d_shiftSeq;
    h_statDescr.tPrmShiftSeq.strides[0] = strides_shiftSeq[0];
    h_statDescr.tPrmShiftSeq.strides[1] = strides_shiftSeq[1];

    // tPrmUnShiftSeq
    h_statDescr.tPrmUnShiftSeq.pAddr      = d_unShiftSeq;
    h_statDescr.tPrmUnShiftSeq.strides[0] = strides_unShiftSeq[0];

    ch_est::puschRxChEstStatDescr_t* d_statDescr;
    CUDA_CHECK(cudaMalloc(&d_statDescr, sizeof(ch_est::puschRxChEstStatDescr_t)));
    CUDA_CHECK(cudaMemcpy(d_statDescr, &h_statDescr,
                          sizeof(ch_est::puschRxChEstStatDescr_t), cudaMemcpyHostToDevice));

    // ----------------------------------------------------------
    // Build cuphyPuschRxUeGrpPrms_t (CPU → GPU)
    // ----------------------------------------------------------
    cuphyPuschRxUeGrpPrms_t h_ueGrpPrms;
    memset(&h_ueGrpPrms, 0, sizeof(h_ueGrpPrms));

    h_ueGrpPrms.nRxAnt      = (uint16_t)nRxAnt;
    h_ueGrpPrms.nLayers     = (uint16_t)nLayers;
    h_ueGrpPrms.nPrb        = (uint16_t)N_PRB;
    h_ueGrpPrms.startPrb    = 0;
    h_ueGrpPrms.slotNum     = (uint16_t)SLOT_NUM;
    h_ueGrpPrms.dmrsMaxLen  = 1;  // single-symbol DMRS
    h_ueGrpPrms.dmrsSymLoc[0] = 2;   // first DMRS symbol
    h_ueGrpPrms.dmrsSymLoc[1] = 11;  // second DMRS symbol
    h_ueGrpPrms.dmrsCnt      = 2;
    h_ueGrpPrms.scid          = 0;
    h_ueGrpPrms.dmrsScrmId    = 0;
    h_ueGrpPrms.nDmrsCdmGrpsNoData = 1;

    // activeDMRSGridBmsk: grid0 active for 1L, both active for 2L/4L
    h_ueGrpPrms.activeDMRSGridBmsk = (nLayers == 1) ? 0x1 : 0x3;

    // OCC masks: single-symbol DMRS, 1 TOCC, active FOCC depending on nLayers
    //   1L: 1 FOCC -> 0x1
    //   2L: 1 FOCC per TOCC (port0 grid0, port1 grid1) -> 0x1 for both grids
    //   4L: 2 FOCCs (port0,1 on grid0 focc0,1; port2,3 on grid1 focc0,1) -> 0x3
    for (int g = 0; g < 2; g++) {
        h_ueGrpPrms.activeTOCCBmsk[g] = 0x1; // 1 TOCC (single-symbol)
        h_ueGrpPrms.activeFOCCBmsk[g] = (nLayers >= 4) ? 0x3 : 0x1;
    }

    // OCC indices: for DMRS Type1, single-TOCC
    // bit[1:0] = FOCC/TOCC index, bit[2] = grid index
    if (nLayers == 1) {
        h_ueGrpPrms.OCCIdx[0] = 0; // Grid 0, OCC 0
    } else if (nLayers == 2) {
        h_ueGrpPrms.OCCIdx[0] = 0; // Grid 0, OCC 0
        h_ueGrpPrms.OCCIdx[1] = 4; // Grid 1, OCC 0
    } else if (nLayers == 4) {
        h_ueGrpPrms.OCCIdx[0] = 0; // Grid 0, OCC 0
        h_ueGrpPrms.OCCIdx[1] = 1; // Grid 0, OCC 1
        h_ueGrpPrms.OCCIdx[2] = 4; // Grid 1, OCC 0
        h_ueGrpPrms.OCCIdx[3] = 5; // Grid 1, OCC 1
    }

    // tInfoDataRx: (NF, ND, nRxAnt)  — cuphyTensorInfo3_t
    h_ueGrpPrms.tInfoDataRx.pAddr      = d_dataRx;
    h_ueGrpPrms.tInfoDataRx.elemType   = CUPHY_C_16F;   // FP16 complex
    h_ueGrpPrms.tInfoDataRx.strides[0] = strides_dataRx[0];
    h_ueGrpPrms.tInfoDataRx.strides[1] = strides_dataRx[1];
    h_ueGrpPrms.tInfoDataRx.strides[2] = strides_dataRx[2];

    // tInfoHEst: (nRxAnt, nLayers, NF, NH=1)  — cuphyTensorInfo4_t
    h_ueGrpPrms.tInfoHEst.pAddr      = d_hEst;
    h_ueGrpPrms.tInfoHEst.elemType   = CUPHY_C_32F;     // FP32 complex
    h_ueGrpPrms.tInfoHEst.strides[0] = strides_hEst[0];
    h_ueGrpPrms.tInfoHEst.strides[1] = strides_hEst[1];
    h_ueGrpPrms.tInfoHEst.strides[2] = strides_hEst[2];
    h_ueGrpPrms.tInfoHEst.strides[3] = strides_hEst[3];

    // tInfoChEstDbg: same shape as tHEst (dummy — not checked)  — cuphyTensorInfo4_t
    h_ueGrpPrms.tInfoChEstDbg.pAddr      = d_chEstDbg;
    h_ueGrpPrms.tInfoChEstDbg.elemType   = CUPHY_C_32F;
    h_ueGrpPrms.tInfoChEstDbg.strides[0] = strides_dbg[0];
    h_ueGrpPrms.tInfoChEstDbg.strides[1] = strides_dbg[1];
    h_ueGrpPrms.tInfoChEstDbg.strides[2] = strides_dbg[2];
    h_ueGrpPrms.tInfoChEstDbg.strides[3] = strides_dbg[3];

    cuphyPuschRxUeGrpPrms_t* d_ueGrpPrms;
    CUDA_CHECK(cudaMalloc(&d_ueGrpPrms, sizeof(cuphyPuschRxUeGrpPrms_t)));
    CUDA_CHECK(cudaMemcpy(d_ueGrpPrms, &h_ueGrpPrms,
                          sizeof(cuphyPuschRxUeGrpPrms_t), cudaMemcpyHostToDevice));

    // ----------------------------------------------------------
    // Build Dynamic Descriptor — we need TWO: one per DMRS time inst
    // chEstTimeInst=0 -> dmrsSymLoc[0] = sym 2
    // chEstTimeInst=1 -> dmrsSymLoc[0] = sym 11
    // ----------------------------------------------------------
    const uint8_t dmrsSymbols[1] = {2};

    ch_est::puschRxChEstDynDescr_t* d_dynDescr[1];
    for (int t = 0; t < 1; t++) {
        ch_est::puschRxChEstDynDescr_t h_dynDescr;
        memset(&h_dynDescr, 0, sizeof(h_dynDescr));

        h_dynDescr.chEstTimeInst       = (uint8_t)t;
        h_dynDescr.pDrvdUeGrpPrms      = d_ueGrpPrms;
        h_dynDescr.hetCfgUeGrpMap[0]   = 0; // UE group 0

        // dmrsSymPos is indexed as pDmrsSymPos = &drvdUeGrpPrms.dmrsSymLoc[chEstTimeInst * dmrsMaxLen]
        // We bake the symbol into dmrsSymLoc[t*1] in the ueGrpPrms.
        // Since dmrsMaxLen=1, chEstTimeInst=t -> dmrsSymLoc[t]
        // We've already set dmrsSymLoc[0]=2, dmrsSymLoc[1]=11 above.

        CUDA_CHECK(cudaMalloc(&d_dynDescr[t], sizeof(ch_est::puschRxChEstDynDescr_t)));
        CUDA_CHECK(cudaMemcpy(d_dynDescr[t], &h_dynDescr,
                              sizeof(ch_est::puschRxChEstDynDescr_t), cudaMemcpyHostToDevice));
    }

    // ----------------------------------------------------------
    // Launch geometry
    //   blockDim = N_DMRS_IN * N_TONES_PER_PRB = 8 * 12 = 96
    //   gridDim  = (ceil(N_PRB/N_DMRS_OUT), nRxAnt, N_UE_GRPS)
    //            = (ceil(273/4), nRxAnt, 1)
    //            = (69, nRxAnt, 1)
    // ----------------------------------------------------------
    const uint32_t N_PRB_CLUSTERS_U32 = (N_PRB + N_DMRS_OUT - 1) / N_DMRS_OUT; // 69
    dim3 blockDim(N_DMRS_IN * N_TONES_PER_PRB_P);                           // 96
    dim3 gridDim(N_PRB_CLUSTERS_U32, (uint32_t)nRxAnt, N_UE_GRPS);             // (69, nRxAnt, 1)

    printf("[LAUNCH] blockDim=(%d,1,1)  gridDim=(%d,%d,%d)\n",
           blockDim.x, gridDim.x, gridDim.y, gridDim.z);
    printf("[LAUNCH] Total threads = %d\n\n",
           blockDim.x * gridDim.x * gridDim.y * gridDim.z);

    // ----------------------------------------------------------
    // Lambda: run one set of warmup + timing for a given dynDescr
    // ----------------------------------------------------------
    auto profile_one = [&](int dmrsSym, ch_est::puschRxChEstDynDescr_t* d_dyn) -> float {
        // Warmup
        for (int i = 0; i < WARMUP_ITERS; i++) {
            if      (nLayers == 1) launch_kernel<1>(d_statDescr, d_dyn, gridDim, blockDim, stream);
            else if (nLayers == 2) launch_kernel<2>(d_statDescr, d_dyn, gridDim, blockDim, stream);
            else if (nLayers == 4) launch_kernel<4>(d_statDescr, d_dyn, gridDim, blockDim, stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaGetLastError());

        // Timing
        cudaEvent_t evStart, evStop;
        CUDA_CHECK(cudaEventCreate(&evStart));
        CUDA_CHECK(cudaEventCreate(&evStop));

        CUDA_CHECK(cudaEventRecord(evStart, stream));
        for (int i = 0; i < TIMING_ITERS; i++) {
            if      (nLayers == 1) launch_kernel<1>(d_statDescr, d_dyn, gridDim, blockDim, stream);
            else if (nLayers == 2) launch_kernel<2>(d_statDescr, d_dyn, gridDim, blockDim, stream);
            else if (nLayers == 4) launch_kernel<4>(d_statDescr, d_dyn, gridDim, blockDim, stream);
        }
        CUDA_CHECK(cudaEventRecord(evStop, stream));
        CUDA_CHECK(cudaEventSynchronize(evStop));
        CUDA_CHECK(cudaGetLastError());

        float elapsedMs = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, evStart, evStop));

        CUDA_CHECK(cudaEventDestroy(evStart));
        CUDA_CHECK(cudaEventDestroy(evStop));

        float avgUs = (elapsedMs * 1000.0f) / (float)TIMING_ITERS;
        return avgUs;
    };

    // ----------------------------------------------------------
    // Run profiling for DMRS sym 2 and sym 11
    // ----------------------------------------------------------
    float latency_us[1];
    for (int t = 0; t < 1; t++) {
        printf("[PROFILE] DMRS symbol index = %d\n", dmrsSymbols[t]);
        latency_us[t] = profile_one(dmrsSymbols[t], d_dynDescr[t]);
        printf("  Avg latency (RTX 4070 Super) : %.3f us\n", latency_us[t]);
    }

    // ----------------------------------------------------------
    // Print summary & GB10 estimate
    // ----------------------------------------------------------
    printf("\n=======================================================\n");
    printf(" RESULTS SUMMARY  [%s]\n", argv[1]);
    printf("=======================================================\n");
    printf(" %-30s %10s %10s\n", "Config", "RTX4070S(us)", "GB10 est.(us)");
    printf(" %-30s %10s %10s\n", "------", "------------", "-------------");

    // Simple compute-bound scaling: GB10 is faster by CUDA core ratio
    double scale = RTX4070S_CUDA_CORES / GB10_CUDA_CORES;  // < 1.0 (GB10 faster)

    for (int t = 0; t < 1; t++) {
        char label[64];
        snprintf(label, sizeof(label), "%s DMRS sym=%d", argv[1], dmrsSymbols[t]);
        double gb10_est = (double)latency_us[t] * scale;
        printf(" %-30s %10.3f %10.3f\n", label, latency_us[t], gb10_est);
    }

    printf("\n[NOTE] GB10 estimate assumes compute-bound scaling:\n");
    printf("       RTX 4070 Super  ~%.0f CUDA cores\n", RTX4070S_CUDA_CORES);
    printf("       NVIDIA Spark GB10 ~%.0f CUDA cores (estimated)\n", GB10_CUDA_CORES);
    printf("       Scale factor: %.3fx (GB10 / RTX4070S)\n\n", GB10_CUDA_CORES / RTX4070S_CUDA_CORES);

    // ----------------------------------------------------------
    // Cleanup
    // ----------------------------------------------------------
    CUDA_CHECK(cudaFree(d_dataRx));
    CUDA_CHECK(cudaFree(d_freqInterp));
    CUDA_CHECK(cudaFree(d_shiftSeq));
    CUDA_CHECK(cudaFree(d_unShiftSeq));
    CUDA_CHECK(cudaFree(d_hEst));
    CUDA_CHECK(cudaFree(d_chEstDbg));
    CUDA_CHECK(cudaFree(d_tmp));
    CUDA_CHECK(cudaFree(d_strides_dataRx));
    CUDA_CHECK(cudaFree(d_strides_freqInterp));
    CUDA_CHECK(cudaFree(d_strides_shiftSeq));
    CUDA_CHECK(cudaFree(d_strides_unShift));
    CUDA_CHECK(cudaFree(d_strides_hEst));
    CUDA_CHECK(cudaFree(d_strides_dbg));
    CUDA_CHECK(cudaFree(d_statDescr));
    CUDA_CHECK(cudaFree(d_ueGrpPrms));
    for (int t = 0; t < 2; t++) CUDA_CHECK(cudaFree(d_dynDescr[t]));
    CURAND_CHECK(curandDestroyGenerator(gen));
    CUDA_CHECK(cudaStreamDestroy(stream));

    printf("[DONE]\n");
    return EXIT_SUCCESS;
}
