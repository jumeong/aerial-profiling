/*
 * Standalone profiling harness for windowedChEstFilterNoDftSOfdmKernel
 *
 * Target configs (nPrb=273, dmrsMaxLen=1):
 *   Config A: nRxAnt=64,  nLayers=2  ->  blockDim=(48,4,1), gridDim=(137,16,1)
 *   Config B: nRxAnt=16,  nLayers=4  ->  blockDim=(96,4,1), gridDim=(137, 4,1)
 *
 * Template instantiation for nPrb=273 (273%4==1, so N_PRB_IN=4, N_INTERP_PRB_OUT=2):
 *   TStorage=float, TCompute=float, TDataRx=float (unused in this kernel)
 *   N_DMRS_GRIDS_PER_PRB=2, N_DMRS_PRB_IN_PER_CLUSTER=4,
 *   N_DMRS_INTERP_PRB_OUT_PER_CLUSTER=2, N_DMRS_SYMS=1
 *
 * Build (example):
 *   nvcc -O3 -arch=sm_90 -std=c++17 bench_windowed_chest.cu -o bench_windowed_chest
 *
 * Profile:
 *   nsys profile --trace=cuda ./bench_windowed_chest
 *   ncu --set full -o chest_ncu ./bench_windowed_chest
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuComplex.h>

using namespace cooperative_groups;
namespace cg = cooperative_groups;

// ---------------------------------------------------------------------------
// Utility macros
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t _e = (call);                                                 \
        if (_e != cudaSuccess) {                                                 \
            fprintf(stderr, "[CUDA error] %s:%d  %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                 \
            exit(1);                                                             \
        }                                                                        \
    } while (0)

#define CUDA_BOTH      __host__ __device__
#define CUDA_BOTH_INLINE __forceinline__ __host__ __device__
#define CUDA_INLINE    __forceinline__ __device__

// ---------------------------------------------------------------------------
// Constants (mirrored from ch_est.cu / cuphy.h)
// ---------------------------------------------------------------------------
static constexpr uint32_t N_TONES_PER_PRB  = 12;
static constexpr uint32_t MAX_N_LAYERS_PUSCH= 8;
static constexpr uint32_t MAX_N_USER_GROUPS_SUPPORTED = 16; // conservative upper bound
static constexpr uint32_t CUPHY_PUSCH_RX_CH_EST_N_DIM_FREQ_INTERP_COEFS = 3;

// ---------------------------------------------------------------------------
// tensor_ref helper (minimal copy from ch_est.cu)
// ---------------------------------------------------------------------------
template <typename TElem>
struct tensor_ref {
    TElem*     pAddr;
    const int* strides;

    CUDA_BOTH tensor_ref(void* pAddr_, const int* pStrides) :
        pAddr(static_cast<TElem*>(pAddr_)), strides(pStrides) {}

    CUDA_BOTH int offset(int i0) const { return strides[0]*i0; }
    CUDA_BOTH int offset(int i0, int i1) const { return strides[0]*i0 + strides[1]*i1; }
    CUDA_BOTH int offset(int i0, int i1, int i2) const {
        return strides[0]*i0 + strides[1]*i1 + strides[2]*i2;
    }
    CUDA_BOTH int offset(int i0, int i1, int i2, int i3) const {
        return strides[0]*i0 + strides[1]*i1 + strides[2]*i2 + strides[3]*i3;
    }

    CUDA_BOTH TElem& operator()(int i0)                         { return *(pAddr + offset(i0)); }
    CUDA_BOTH TElem& operator()(int i0, int i1)                 { return *(pAddr + offset(i0,i1)); }
    CUDA_BOTH TElem& operator()(int i0, int i1, int i2)         { return *(pAddr + offset(i0,i1,i2)); }
    CUDA_BOTH TElem& operator()(int i0, int i1, int i2, int i3) { return *(pAddr + offset(i0,i1,i2,i3)); }

    CUDA_BOTH const TElem& operator()(int i0) const                         { return *(pAddr + offset(i0)); }
    CUDA_BOTH const TElem& operator()(int i0, int i1) const                 { return *(pAddr + offset(i0,i1)); }
    CUDA_BOTH const TElem& operator()(int i0, int i1, int i2) const         { return *(pAddr + offset(i0,i1,i2)); }
    CUDA_BOTH const TElem& operator()(int i0, int i1, int i2, int i3) const { return *(pAddr + offset(i0,i1,i2,i3)); }
};

// ---------------------------------------------------------------------------
// type_convert: identity for float->float (extend for fp16 if needed)
// ---------------------------------------------------------------------------
template <typename Tout, typename Tin>
__forceinline__ __device__ __host__ Tout type_convert(Tin val) { return static_cast<Tout>(val); }

template <>
__forceinline__ __device__ __host__ cuComplex type_convert<cuComplex, cuComplex>(cuComplex val) { return val; }

// ---------------------------------------------------------------------------
// complex_from_scalar: for TCompute=float -> cuComplex
// ---------------------------------------------------------------------------
template <typename T> struct complex_from_scalar;
template <> struct complex_from_scalar<float>  { typedef cuComplex type; };
// Add __half -> __half2 here if you switch to fp16

// ---------------------------------------------------------------------------
// Minimal tensor info types (only pAddr + strides, no elemType needed on GPU)
// ---------------------------------------------------------------------------
struct bench_TensorInfo4 {
    void*   pAddr;
    int32_t strides[4];
};

struct bench_TensorInfo3 {
    void*   pAddr;
    int32_t strides[3];
};

// ---------------------------------------------------------------------------
// Minimal tensor param (from ch_est_types.hpp puschRxChEstTensorPrm_t)
// ---------------------------------------------------------------------------
template <size_t NDim>
struct bench_ChEstTensorPrm {
    void* pAddr;
    int   strides[NDim];
};

// ---------------------------------------------------------------------------
// Minimal UE group params (only fields accessed by windowedChEstFilterNoDftSOfdmKernel)
// ---------------------------------------------------------------------------
struct bench_UeGrpPrms {
    uint16_t nPrb;
    uint16_t nRxAnt;
    uint16_t nLayers;
    uint8_t  nDmrsCdmGrpsNoData;
    uint8_t  OCCIdx_data[MAX_N_LAYERS_PUSCH]; // owned storage
    uint8_t* OCCIdx;                          // pointer used by kernel
    // tInfoDmrsLSEst: [tone_in_cluster, layer, ant, time]
    bench_TensorInfo4 tInfoDmrsLSEst;
    // tInfoHEst: [ant, layer, abs_tone, time]
    bench_TensorInfo4 tInfoHEst;
};

// ---------------------------------------------------------------------------
// Minimal static descriptor (only freq interp coef tensors)
// ---------------------------------------------------------------------------
struct bench_StatDescr {
    // Used when nPrb>=8 && nPrb%4==0  (49x48x3)
    bench_ChEstTensorPrm<3> tPrmFreqInterpCoefs;
    // Used when 3<nPrb<8 or nPrb%4!=0  (25x24x3)  <-- OUR CASE (nPrb=273, 273%4=1)
    bench_ChEstTensorPrm<3> tPrmFreqInterpCoefs4;
    // Used when nPrb<=3               (37x18x3)
    bench_ChEstTensorPrm<3> tPrmFreqInterpCoefsSmall;
};

// ---------------------------------------------------------------------------
// Minimal dynamic descriptor
// ---------------------------------------------------------------------------
struct bench_DynDescr {
    uint8_t          chEstTimeInst;
    bench_UeGrpPrms* pDrvdUeGrpPrms;               // device pointer
    uint32_t         hetCfgUeGrpMap[MAX_N_USER_GROUPS_SUPPORTED];
};

// ---------------------------------------------------------------------------
// Helper inline functions (from ch_est.cu)
// ---------------------------------------------------------------------------
CUDA_INLINE constexpr uint32_t get_inter_dmrs_grid_freq_shift(const uint32_t nDmrsGridsPerPrb)
{
    return (2 == nDmrsGridsPerPrb) ? 1u : 2u;
}

CUDA_INLINE constexpr uint32_t get_inter_dmrs_grid_freq_shift_idx(const uint32_t nDmrsGridsPerPrb,
                                                                    const uint32_t gridIdx)
{
    return ((nDmrsGridsPerPrb - 1) - gridIdx) * get_inter_dmrs_grid_freq_shift(nDmrsGridsPerPrb);
}

CUDA_BOTH_INLINE uint32_t div_round_up(uint32_t val, uint32_t divide_by)
{
    return (val + divide_by - 1) / divide_by;
}

// ---------------------------------------------------------------------------
// The kernel (direct copy from ch_est.cu, structs replaced with bench_ versions)
//
// Template params:
//   TStorage  : scalar type for H output  (float for this bench)
//   TDataRx   : unused in this kernel     (float placeholder)
//   TCompute  : computation type          (float for this bench)
//   N_DMRS_GRIDS_PER_PRB             = 2  (type 1 DMRS)
//   N_DMRS_PRB_IN_PER_CLUSTER        = 4  (273%4!=0 -> else branch)
//   N_DMRS_INTERP_PRB_OUT_PER_CLUSTER= 2
//   N_DMRS_SYMS                      = 1  (dmrsMaxLen=1)
// ---------------------------------------------------------------------------
template <typename TStorage,
          typename TDataRx,
          typename TCompute,
          uint32_t N_DMRS_GRIDS_PER_PRB,
          uint32_t N_DMRS_PRB_IN_PER_CLUSTER,
          uint32_t N_DMRS_INTERP_PRB_OUT_PER_CLUSTER,
          uint32_t N_DMRS_SYMS>
static __device__ void
windowedChEstFilterNoDftSOfdmKernel(
    bench_StatDescr*  pStatDescr,
    bench_DynDescr*   pDynDescr,
    typename complex_from_scalar<TCompute>::type* sh_shiftSeq,
    typename complex_from_scalar<TCompute>::type* sh_unShiftSeq,
    typename complex_from_scalar<TCompute>::type* sh_ls_est)
{
    typedef typename complex_from_scalar<TCompute>::type  TComplexCompute;
    typedef typename complex_from_scalar<TStorage>::type  TComplexStorage;

    // Only type 1 DMRS grids (every other tone) are supported
    constexpr uint32_t N_DMRS_TYPE1_GRIDS_PER_PRB = 2;
    static_assert(N_DMRS_GRIDS_PER_PRB == N_DMRS_TYPE1_GRIDS_PER_PRB,
                  "Kernel only supports type 1 DMRS grids");
    constexpr uint32_t N_DMRS_TONE_STRIDE = 2; // kept for documentation

    thread_block const& block = this_thread_block();

    bench_StatDescr& statDescr = *pStatDescr;
    bench_DynDescr&  dynDescr  = *pDynDescr;

    const uint32_t UE_GRP_IDX    = dynDescr.hetCfgUeGrpMap[blockIdx.z];
    const uint32_t PRB_CLUSTER_IDX = blockIdx.x;

    constexpr uint32_t N_DMRS_INTERP_TONES_PER_GRID = N_TONES_PER_PRB;
    constexpr uint32_t N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER =
        N_DMRS_INTERP_TONES_PER_GRID * N_DMRS_INTERP_PRB_OUT_PER_CLUSTER;

    const uint32_t THREAD_IDX         = threadIdx.x;
    const uint32_t LAYER_IDX          = THREAD_IDX / N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER;
    const uint32_t THREAD_IDX_MOD_LAYER = THREAD_IDX - LAYER_IDX * N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER;

    const uint32_t tid      = threadIdx.x;
    const uint32_t nthreads = blockDim.x;

    bench_UeGrpPrms& drvdUeGrpPrms = pDynDescr->pDrvdUeGrpPrms[UE_GRP_IDX];

    const uint16_t nPrb              = drvdUeGrpPrms.nPrb;
    const uint16_t nRxAnt            = drvdUeGrpPrms.nRxAnt;
    uint8_t*       OCCIdx            = drvdUeGrpPrms.OCCIdx;
    const uint8_t  nDmrsCdmGrpsNoData = drvdUeGrpPrms.nDmrsCdmGrpsNoData;

    const uint32_t BS_ANT_IDX = blockDim.y * blockIdx.y + threadIdx.y;
    const uint32_t N_PRB_CLUSTERS_PER_BS_ANT =
        div_round_up(nPrb, static_cast<uint16_t>(N_DMRS_INTERP_PRB_OUT_PER_CLUSTER));
    if ((PRB_CLUSTER_IDX >= N_PRB_CLUSTERS_PER_BS_ANT) || (BS_ANT_IDX >= nRxAnt)) return;

    const uint32_t N_PRB_CLUSTERS = N_PRB_CLUSTERS_PER_BS_ANT;
    const uint16_t nLayers        = drvdUeGrpPrms.nLayers;
    const uint8_t  chEstTimeInst  = dynDescr.chEstTimeInst;

    const uint32_t N_EDGE_PRB =
        (N_DMRS_PRB_IN_PER_CLUSTER - N_DMRS_INTERP_PRB_OUT_PER_CLUSTER) / 2;

    constexpr uint32_t N_DMRS_GRID_TONES_PER_PRB =
        N_TONES_PER_PRB / N_DMRS_GRIDS_PER_PRB;
    constexpr uint32_t N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER =
        N_DMRS_GRID_TONES_PER_PRB * N_DMRS_PRB_IN_PER_CLUSTER;

    // Select frequency interpolation coefficients based on nPrb
    tensor_ref<const TCompute> tFreqInterpCoefs = [nPrb, &statDescr]() -> auto {
        if (nPrb <= 3) {
            return tensor_ref<const TCompute>{
                statDescr.tPrmFreqInterpCoefsSmall.pAddr,
                statDescr.tPrmFreqInterpCoefsSmall.strides};
        } else if (nPrb >= 8 && nPrb % 4 == 0) {
            return tensor_ref<const TCompute>{
                statDescr.tPrmFreqInterpCoefs.pAddr,
                statDescr.tPrmFreqInterpCoefs.strides};
        } else {
            return tensor_ref<const TCompute>{
                statDescr.tPrmFreqInterpCoefs4.pAddr,
                statDescr.tPrmFreqInterpCoefs4.strides};
        }
    }();

    const uint32_t filtIdx = [nPrb, PRB_CLUSTER_IDX, N_PRB_CLUSTERS]() -> uint32_t {
        if (nPrb == 0) {
            return 0;
        } else if (nPrb <= 3) {
            return nPrb - 1;
        } else {
            constexpr uint32_t MIDDLE_INTERP_FILT_IDX     = 0;
            constexpr uint32_t LOWER_EDGE_INTERP_FILT_IDX = 1;
            constexpr uint32_t UPPER_EDGE_INTERP_FILT_IDX = 2;
            if (PRB_CLUSTER_IDX == 0)
                return LOWER_EDGE_INTERP_FILT_IDX;
            else if (PRB_CLUSTER_IDX == N_PRB_CLUSTERS - 1)
                return UPPER_EDGE_INTERP_FILT_IDX;
            else
                return MIDDLE_INTERP_FILT_IDX;
        }
    }();

    // First PRB in the cluster
    uint32_t prbClusterStartIdx =
        (PRB_CLUSTER_IDX * N_DMRS_INTERP_PRB_OUT_PER_CLUSTER) - N_EDGE_PRB;
    if (0 == PRB_CLUSTER_IDX)
        prbClusterStartIdx = 0;
    if ((N_PRB_CLUSTERS - 1) == PRB_CLUSTER_IDX)
        prbClusterStartIdx = nPrb - N_DMRS_PRB_IN_PER_CLUSTER;

    const uint32_t toneOffset           = prbClusterStartIdx * N_TONES_PER_PRB;
    const uint32_t DMRS_LS_EST_TONE_OFFSET = toneOffset / N_DMRS_TONE_STRIDE;

    const uint32_t NH_IDX = chEstTimeInst;

    tensor_ref<const TComplexCompute> tInfoDmrsLSEst(
        drvdUeGrpPrms.tInfoDmrsLSEst.pAddr,
        drvdUeGrpPrms.tInfoDmrsLSEst.strides);
    const TComplexCompute* dmrsLSEst =
        tInfoDmrsLSEst.pAddr +
        tInfoDmrsLSEst.offset(DMRS_LS_EST_TONE_OFFSET, 0, BS_ANT_IDX, NH_IDX);
    const uint32_t toneStride  = tInfoDmrsLSEst.strides[0];
    const uint32_t layerStride = tInfoDmrsLSEst.strides[1];

    // Load LS estimates into shared memory, applying shift sequence
    const int nIter = N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER * nLayers;
    for (int i = tid; i < nIter; i += nthreads) {
        const int layer = i / N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER;
        const int tone  = i % N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER;
        sh_ls_est[i]    = dmrsLSEst[tone * toneStride + layer * layerStride] * sh_shiftSeq[tone];
    }

    __syncthreads();

    // Threads beyond nLayers have no outputs
    if (LAYER_IDX >= nLayers) return;

    const uint32_t DMRS_INTERP_TONE_IDX = THREAD_IDX_MOD_LAYER % N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER;

    constexpr uint32_t HALF_N_EDGE_TONES =
        N_TONES_PER_PRB * (N_DMRS_PRB_IN_PER_CLUSTER - N_DMRS_INTERP_PRB_OUT_PER_CLUSTER) / 2;
    uint32_t CLUSTER_INTERP_TONE_IDX = DMRS_INTERP_TONE_IDX + HALF_N_EDGE_TONES;
    if (0 == PRB_CLUSTER_IDX)
        CLUSTER_INTERP_TONE_IDX -= HALF_N_EDGE_TONES;
    if ((N_PRB_CLUSTERS - 1) == PRB_CLUSTER_IDX)
        CLUSTER_INTERP_TONE_IDX += HALF_N_EDGE_TONES;

    const uint32_t INTERP_PRB_CLUSTER_IDX = blockIdx.x;
    uint32_t INTERP_DMRS_ABS_TONE_IDX =
        INTERP_PRB_CLUSTER_IDX * N_DMRS_INTERP_PRB_OUT_PER_CLUSTER * N_TONES_PER_PRB +
        DMRS_INTERP_TONE_IDX;

    if ((N_DMRS_PRB_IN_PER_CLUSTER == 4) && (nPrb % 2 == 1) &&
        (PRB_CLUSTER_IDX == N_PRB_CLUSTERS - 1)) {
        INTERP_DMRS_ABS_TONE_IDX -= N_TONES_PER_PRB;
    }
    const uint32_t MAX_INTERP_ABS_TONE = nPrb * N_TONES_PER_PRB - 1;

    if (INTERP_DMRS_ABS_TONE_IDX <= MAX_INTERP_ABS_TONE) {
        const TCompute scaling =
            (nDmrsCdmGrpsNoData == 1) ? static_cast<TCompute>(1.414213562373095f) : 1.0f;
        const uint32_t gridShiftIdx =
            get_inter_dmrs_grid_freq_shift_idx(N_DMRS_GRIDS_PER_PRB,
                                               (OCCIdx[LAYER_IDX] >> 2) & 0x1);

        tensor_ref<TComplexStorage> tHEst(drvdUeGrpPrms.tInfoHEst.pAddr,
                                          drvdUeGrpPrms.tInfoHEst.strides);
        TComplexStorage* HEst =
            tHEst.pAddr + tHEst.offset(BS_ANT_IDX, 0, INTERP_DMRS_ABS_TONE_IDX, NH_IDX);
        const uint32_t estLayerStride = tHEst.strides[1];

        const TCompute* coefs =
            tFreqInterpCoefs.pAddr +
            tFreqInterpCoefs.offset(DMRS_INTERP_TONE_IDX + gridShiftIdx, 0, filtIdx);
        const int coefToneStride = tFreqInterpCoefs.strides[1];

        TComplexCompute accum{0.0f, 0.0f};
        for (int j = 0; j < static_cast<int>(N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER); j++) {
            // accum += sh_ls_est[...] * coefs[...]  (complex multiply-accumulate)
            const TComplexCompute ls  = sh_ls_est[LAYER_IDX * N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER + j];
            const TCompute        c   = coefs[j * coefToneStride];
            accum.x += ls.x * c;
            accum.y += ls.y * c;
        }

        // accum *= sh_unShiftSeq[...] * scaling  (mirrors original: accum *= (unShift * scaling))
        const TComplexCompute unshift = sh_unShiftSeq[CLUSTER_INTERP_TONE_IDX + gridShiftIdx];
        const float ax = accum.x, ay = accum.y;
        accum.x = (ax * unshift.x - ay * unshift.y) * scaling;
        accum.y = (ax * unshift.y + ay * unshift.x) * scaling;

        HEst[LAYER_IDX * estLayerStride] = type_convert<TComplexStorage>(accum);
    }
} // windowedChEstFilterNoDftSOfdmKernel

// ---------------------------------------------------------------------------
// Wrapper __global__ kernel
//
// Shared memory layout (from dispatch kernel):
//   [ sh_shiftSeq   : MAX_NUM_PRBS_PER_FILTER * (N_TONES_PER_PRB/N_DMRS_GRIDS_PER_PRB) ]
//   [ sh_unShiftSeq : (MAX_NUM_PRBS_PER_FILTER * N_TONES_PER_PRB) + 1                  ]
//   [ sh_ls_est_full: nRxAntPerBlock * nMaxLayers * MAX_N_TOTAL_DMRS_GRID_TONES         ] (extern)
// ---------------------------------------------------------------------------
template <typename TStorage,
          typename TDataRx,
          typename TCompute,
          uint32_t N_DMRS_GRIDS_PER_PRB,
          uint32_t N_DMRS_PRB_IN_PER_CLUSTER,
          uint32_t N_DMRS_INTERP_PRB_OUT_PER_CLUSTER,
          uint32_t N_DMRS_SYMS>
__global__ void
bench_windowedChEstFilterWrapper(bench_StatDescr* pStatDescr, bench_DynDescr* pDynDescr)
{
    typedef typename complex_from_scalar<TCompute>::type TComplexCompute;

    constexpr uint32_t MAX_NUM_PRBS_PER_FILTER = 8;
    constexpr uint32_t N_DMRS_GRID_TONES_PER_PRB = N_TONES_PER_PRB / N_DMRS_GRIDS_PER_PRB;
    constexpr uint32_t MAX_N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER =
        N_DMRS_GRID_TONES_PER_PRB * MAX_NUM_PRBS_PER_FILTER;

    // Static shared memory for shift/unshift sequences
    __shared__ TComplexCompute sh_shiftSeq[MAX_NUM_PRBS_PER_FILTER * N_DMRS_GRID_TONES_PER_PRB];
    __shared__ TComplexCompute sh_unShiftSeq[(MAX_NUM_PRBS_PER_FILTER * N_TONES_PER_PRB) + 1];

    // Dynamic shared memory for LS estimates
    extern __shared__ TComplexCompute sh_ls_est_full[];

    // Initialize shift/unshift sequences to identity (no shift = delay_mean=0)
    // In production, these are computed from the measured delay mean.
    const uint32_t tid      = threadIdx.x + threadIdx.y * blockDim.x;
    const uint32_t nthreads = blockDim.x * blockDim.y;

    const uint32_t nShift   = MAX_NUM_PRBS_PER_FILTER * N_DMRS_GRID_TONES_PER_PRB;
    const uint32_t nUnShift = (MAX_NUM_PRBS_PER_FILTER * N_TONES_PER_PRB) + 1;

    for (uint32_t i = tid; i < nShift; i += nthreads) {
        sh_shiftSeq[i] = TComplexCompute{1.0f, 0.0f};   // exp(j*0) = 1+j0
    }
    for (uint32_t i = tid; i < nUnShift; i += nthreads) {
        sh_unShiftSeq[i] = TComplexCompute{1.0f, 0.0f};
    }

    __syncthreads();

    // Each thread block handles one antenna slice of sh_ls_est
    TComplexCompute* sh_ls_est = sh_ls_est_full +
        threadIdx.y * pDynDescr->pDrvdUeGrpPrms[0].nLayers * MAX_N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER;

    windowedChEstFilterNoDftSOfdmKernel<
        TStorage, TDataRx, TCompute,
        N_DMRS_GRIDS_PER_PRB,
        N_DMRS_PRB_IN_PER_CLUSTER,
        N_DMRS_INTERP_PRB_OUT_PER_CLUSTER,
        N_DMRS_SYMS>(pStatDescr, pDynDescr, sh_shiftSeq, sh_unShiftSeq, sh_ls_est);
}

// ---------------------------------------------------------------------------
// CUCOMPLEX operator overloads (needed for accum += prod etc.)
// ---------------------------------------------------------------------------
__device__ __forceinline__ cuComplex operator+(cuComplex a, cuComplex b) {
    return make_cuComplex(a.x + b.x, a.y + b.y);
}
__device__ __forceinline__ cuComplex operator+=(cuComplex& a, cuComplex b) {
    a.x += b.x; a.y += b.y; return a;
}
__device__ __forceinline__ cuComplex operator*(cuComplex a, float s) {
    return make_cuComplex(a.x * s, a.y * s);
}
__device__ __forceinline__ cuComplex operator*(cuComplex a, cuComplex b) {
    return make_cuComplex(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

// ---------------------------------------------------------------------------
// Host-side random fill helpers
// ---------------------------------------------------------------------------
static void fill_random_cuComplex(cuComplex* p, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        p[i].x = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        p[i].y = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
}

static void fill_identity_real(float* p, size_t n)
{
    // Simple non-zero coefficients for filter (identity-like)
    for (size_t i = 0; i < n; ++i)
        p[i] = 1.0f / 24.0f; // normalize by number of input tones
}

// ---------------------------------------------------------------------------
// Profiling config descriptor
// ---------------------------------------------------------------------------
struct BenchConfig {
    const char* name;
    uint16_t nPrb;
    uint16_t nRxAnt;
    uint16_t nLayers;
    uint8_t  dmrsMaxLen;
    int      nWarmup;
    int      nIter;
};

// ---------------------------------------------------------------------------
// Run one configuration
// ---------------------------------------------------------------------------
void run_config(const BenchConfig& cfg)
{
    printf("=== Config: %s ===\n", cfg.name);
    printf("  nPrb=%u, nRxAnt=%u, nLayers=%u, dmrsMaxLen=%u\n",
           cfg.nPrb, cfg.nRxAnt, cfg.nLayers, cfg.dmrsMaxLen);

    // -----------------------------------------------------------------------
    // Derived parameters for nPrb=273 (273%4==1 -> N_PRB_IN=4, N_INTERP_OUT=2)
    // -----------------------------------------------------------------------
    constexpr uint32_t N_DMRS_GRIDS_PER_PRB             = 2;
    constexpr uint32_t N_DMRS_PRB_IN_PER_CLUSTER        = 4;
    constexpr uint32_t N_DMRS_INTERP_PRB_OUT_PER_CLUSTER= 2;
    constexpr uint32_t N_DMRS_SYMS                      = 1; // dmrsMaxLen=1

    constexpr uint32_t N_TONES                          = N_TONES_PER_PRB;
    constexpr uint32_t N_DMRS_GRID_TONES_PER_PRB        = N_TONES / N_DMRS_GRIDS_PER_PRB; // 6
    constexpr uint32_t N_TOTAL_DMRS_GRID_TONES          = N_DMRS_GRID_TONES_PER_PRB * N_DMRS_PRB_IN_PER_CLUSTER; // 24
    constexpr uint32_t N_TOTAL_DMRS_INTERP_TONES        = N_TONES * N_DMRS_INTERP_PRB_OUT_PER_CLUSTER;           // 24
    constexpr uint32_t MAX_N_TOTAL_DMRS_GRID_PER_CLUSTER= N_DMRS_GRID_TONES_PER_PRB * 8;                         // 48

    // Launch config computation (matches production dispatch logic)
    const uint32_t nPrbClusters = (cfg.nPrb + N_DMRS_INTERP_PRB_OUT_PER_CLUSTER - 1)
                                  / N_DMRS_INTERP_PRB_OUT_PER_CLUSTER;   // ceil(273/2)=137
    const uint32_t nThreadsX    = N_DMRS_INTERP_PRB_OUT_PER_CLUSTER * N_TONES * cfg.nLayers;

    // Antenna batching (mirrors production heuristic: nRxAntPerBlock up to 4)
    const uint32_t LARGE_GRID_SIZE = 256;
    const float targetBlockSz = (float)cfg.nRxAnt * (float)nPrbClusters / (float)LARGE_GRID_SIZE;
    uint32_t nRxAntPerBlock = 1;
    if      (targetBlockSz >= 4.0f && nThreadsX < 128) nRxAntPerBlock = 4;
    else if (targetBlockSz >= 2.0f && nThreadsX < 256) nRxAntPerBlock = 2;

    const dim3 block(nThreadsX, nRxAntPerBlock, 1);
    const dim3 grid(nPrbClusters,
                    (cfg.nRxAnt + nRxAntPerBlock - 1) / nRxAntPerBlock,
                    1 /* 1 UE group */);

    // Shared memory: nRxAntPerBlock * nLayers * MAX_N_TOTAL_DMRS_GRID per cluster
    const size_t smemBytes = sizeof(cuComplex) * nRxAntPerBlock * cfg.nLayers * MAX_N_TOTAL_DMRS_GRID_PER_CLUSTER;

    printf("  Launch: grid=(%u,%u,%u)  block=(%u,%u,%u)  smem=%zu B\n",
           grid.x, grid.y, grid.z, block.x, block.y, block.z, smemBytes);

    // -----------------------------------------------------------------------
    // Allocate and fill tensors
    // -----------------------------------------------------------------------

    // Frequency interp coefficients (tPrmFreqInterpCoefs4: 25 x 24 x 3)
    // For nPrb=273 (273%4!=0) -> tPrmFreqInterpCoefs4 is selected
    // dim: [N_TOTAL_DMRS_INTERP_TONES+gridShift, N_TOTAL_DMRS_GRID_TONES, 3_filters]
    //      [25, 24, 3]
    constexpr int COEF4_DIM0 = 25; // N_INTERP_TONES(24) + inter_grid_shift(1)
    constexpr int COEF4_DIM1 = 24; // N_TOTAL_DMRS_GRID_TONES (6*4)
    constexpr int COEF4_DIM2 = 3;  // 3 filters
    const size_t coef4_n = COEF4_DIM0 * COEF4_DIM1 * COEF4_DIM2;

    float* h_coefs4 = new float[coef4_n];
    fill_identity_real(h_coefs4, coef4_n);
    float* d_coefs4;
    CUDA_CHECK(cudaMalloc(&d_coefs4, coef4_n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_coefs4, h_coefs4, coef4_n * sizeof(float), cudaMemcpyHostToDevice));
    delete[] h_coefs4;

    // Also allocate coefs for the 8-PRB case (49x48x3) for stat descr completeness
    constexpr int COEF_DIM0 = 49, COEF_DIM1 = 48, COEF_DIM2 = 3;
    float* d_coefs;
    CUDA_CHECK(cudaMalloc(&d_coefs, COEF_DIM0*COEF_DIM1*COEF_DIM2*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_coefs, 0, COEF_DIM0*COEF_DIM1*COEF_DIM2*sizeof(float)));

    // LS estimates: [total_dmrs_tones, nLayers, nRxAnt, 1]
    // total_dmrs_tones = nPrb * N_TONES / N_DMRS_GRIDS_PER_PRB = 273*6 = 1638
    const uint32_t total_dmrs_tones = cfg.nPrb * N_DMRS_GRID_TONES_PER_PRB;
    const size_t ls_n = (size_t)total_dmrs_tones * cfg.nLayers * cfg.nRxAnt * 1;
    cuComplex* h_ls = new cuComplex[ls_n];
    fill_random_cuComplex(h_ls, ls_n);
    cuComplex* d_ls;
    CUDA_CHECK(cudaMalloc(&d_ls, ls_n * sizeof(cuComplex)));
    CUDA_CHECK(cudaMemcpy(d_ls, h_ls, ls_n * sizeof(cuComplex), cudaMemcpyHostToDevice));
    delete[] h_ls;

    // H estimate output: [nRxAnt, nLayers, nPrb*12, 1]
    const size_t hest_n = (size_t)cfg.nRxAnt * cfg.nLayers * cfg.nPrb * N_TONES * 1;
    cuComplex* d_hest;
    CUDA_CHECK(cudaMalloc(&d_hest, hest_n * sizeof(cuComplex)));
    CUDA_CHECK(cudaMemset(d_hest, 0, hest_n * sizeof(cuComplex)));

    // OCCIdx: layer -> OCC index mapping (simple: layer i uses grid i%2)
    uint8_t h_occIdx[MAX_N_LAYERS_PUSCH];
    for (int i = 0; i < MAX_N_LAYERS_PUSCH; ++i)
        h_occIdx[i] = (uint8_t)(i & 0x3); // bits[1:0]=fOCC, bit[2]=tOCC, bits[4:3]=grid
    uint8_t* d_occIdx;
    CUDA_CHECK(cudaMalloc(&d_occIdx, MAX_N_LAYERS_PUSCH * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemcpy(d_occIdx, h_occIdx, MAX_N_LAYERS_PUSCH * sizeof(uint8_t), cudaMemcpyHostToDevice));

    // -----------------------------------------------------------------------
    // Build static descriptor (CPU, then copy to GPU)
    // -----------------------------------------------------------------------
    bench_StatDescr h_stat;
    memset(&h_stat, 0, sizeof(h_stat));

    // tPrmFreqInterpCoefs4: strides for [25][24][3] stored row-major
    // offset(i0,i1,i2) = strides[0]*i0 + strides[1]*i1 + strides[2]*i2
    // stride[2]=1, stride[1]=3, stride[0]=24*3=72
    h_stat.tPrmFreqInterpCoefs4.pAddr      = d_coefs4;
    h_stat.tPrmFreqInterpCoefs4.strides[0] = COEF4_DIM1 * COEF4_DIM2; // 72
    h_stat.tPrmFreqInterpCoefs4.strides[1] = COEF4_DIM2;               // 3
    h_stat.tPrmFreqInterpCoefs4.strides[2] = 1;

    // tPrmFreqInterpCoefs (not used for nPrb=273 but must be non-null)
    h_stat.tPrmFreqInterpCoefs.pAddr      = d_coefs;
    h_stat.tPrmFreqInterpCoefs.strides[0] = COEF_DIM1 * COEF_DIM2;
    h_stat.tPrmFreqInterpCoefs.strides[1] = COEF_DIM2;
    h_stat.tPrmFreqInterpCoefs.strides[2] = 1;

    h_stat.tPrmFreqInterpCoefsSmall.pAddr = d_coefs; // placeholder
    h_stat.tPrmFreqInterpCoefsSmall.strides[0] = 1;
    h_stat.tPrmFreqInterpCoefsSmall.strides[1] = 1;
    h_stat.tPrmFreqInterpCoefsSmall.strides[2] = 1;

    bench_StatDescr* d_stat;
    CUDA_CHECK(cudaMalloc(&d_stat, sizeof(bench_StatDescr)));
    CUDA_CHECK(cudaMemcpy(d_stat, &h_stat, sizeof(bench_StatDescr), cudaMemcpyHostToDevice));

    // -----------------------------------------------------------------------
    // Build UE group params (CPU, then copy to GPU)
    // -----------------------------------------------------------------------
    bench_UeGrpPrms h_ueGrp;
    memset(&h_ueGrp, 0, sizeof(h_ueGrp));
    h_ueGrp.nPrb              = cfg.nPrb;
    h_ueGrp.nRxAnt            = cfg.nRxAnt;
    h_ueGrp.nLayers           = cfg.nLayers;
    h_ueGrp.nDmrsCdmGrpsNoData = 2; // typical value; 1 would apply sqrt(2) scaling
    h_ueGrp.OCCIdx            = d_occIdx; // device pointer (filled after struct copy)

    // tInfoDmrsLSEst: [tone, layer, ant, time]
    // strides: tone=nLayers*nRxAnt*1, layer=nRxAnt*1, ant=1, time=1
    h_ueGrp.tInfoDmrsLSEst.pAddr      = d_ls;
    h_ueGrp.tInfoDmrsLSEst.strides[0] = (int32_t)(cfg.nLayers * cfg.nRxAnt); // tone stride
    h_ueGrp.tInfoDmrsLSEst.strides[1] = (int32_t)(cfg.nRxAnt);               // layer stride
    h_ueGrp.tInfoDmrsLSEst.strides[2] = 1;                                    // ant stride
    h_ueGrp.tInfoDmrsLSEst.strides[3] = 1;                                    // time stride

    // tInfoHEst: [ant, layer, tone, time]
    // strides: ant=nLayers*nPrb*12, layer=nPrb*12, tone=1, time=1
    h_ueGrp.tInfoHEst.pAddr      = d_hest;
    h_ueGrp.tInfoHEst.strides[0] = (int32_t)(cfg.nLayers * cfg.nPrb * N_TONES); // ant stride
    h_ueGrp.tInfoHEst.strides[1] = (int32_t)(cfg.nPrb * N_TONES);               // layer stride
    h_ueGrp.tInfoHEst.strides[2] = 1;                                            // tone stride
    h_ueGrp.tInfoHEst.strides[3] = 1;                                            // time stride

    bench_UeGrpPrms* d_ueGrp;
    CUDA_CHECK(cudaMalloc(&d_ueGrp, sizeof(bench_UeGrpPrms)));
    CUDA_CHECK(cudaMemcpy(d_ueGrp, &h_ueGrp, sizeof(bench_UeGrpPrms), cudaMemcpyHostToDevice));

    // -----------------------------------------------------------------------
    // Build dynamic descriptor (CPU, then copy to GPU)
    // -----------------------------------------------------------------------
    bench_DynDescr h_dyn;
    memset(&h_dyn, 0, sizeof(h_dyn));
    h_dyn.chEstTimeInst       = 0;
    h_dyn.pDrvdUeGrpPrms      = d_ueGrp; // device pointer
    h_dyn.hetCfgUeGrpMap[0]  = 0;        // blockIdx.z=0 -> UE group 0

    bench_DynDescr* d_dyn;
    CUDA_CHECK(cudaMalloc(&d_dyn, sizeof(bench_DynDescr)));
    CUDA_CHECK(cudaMemcpy(d_dyn, &h_dyn, sizeof(bench_DynDescr), cudaMemcpyHostToDevice));

    // -----------------------------------------------------------------------
    // IO size report
    // -----------------------------------------------------------------------
    // Input 1: LS estimates — tInfoDmrsLSEst [total_dmrs_tones x nLayers x nRxAnt x 1]
    //   total_dmrs_tones = nPrb * (N_TONES_PER_PRB / N_DMRS_GRIDS_PER_PRB)
    //                    = 273 * 6 = 1638
    //   Each element: cuComplex = 8 bytes
    const size_t bytes_ls_input = ls_n * sizeof(cuComplex);

    // Input 2: Frequency interpolation coefficients — tPrmFreqInterpCoefs4 [25 x 24 x 3]
    //   Each element: float = 4 bytes
    //   (Read-only; typically cached in L2 after first access)
    const size_t bytes_coef_input = (size_t)COEF4_DIM0 * COEF4_DIM1 * COEF4_DIM2 * sizeof(float);

    // Output: H estimates — tInfoHEst [nRxAnt x nLayers x nPrb*12 x 1]
    //   Each element: cuComplex = 8 bytes
    const size_t bytes_hest_output = hest_n * sizeof(cuComplex);

    const size_t bytes_total_io = bytes_ls_input + bytes_coef_input + bytes_hest_output;

    printf("\n  [IO Sizes]\n");
    printf("  Input  LS estimates  (tInfoDmrsLSEst)   : %10zu bytes  = %.3f MB\n",
           bytes_ls_input, bytes_ls_input / 1048576.0);
    printf("  Input  Interp coefs  (tPrmFreqInterpCoefs4, cached): %6zu bytes  = %.3f KB\n",
           bytes_coef_input, bytes_coef_input / 1024.0);
    printf("  Output H estimates   (tInfoHEst)         : %10zu bytes  = %.3f MB\n",
           bytes_hest_output, bytes_hest_output / 1048576.0);
    printf("  Total IO (excl. cached coefs)            : %10zu bytes  = %.3f MB\n",
           bytes_ls_input + bytes_hest_output,
           (bytes_ls_input + bytes_hest_output) / 1048576.0);
    printf("  Total IO (incl. coefs)                   : %10zu bytes  = %.3f MB\n\n",
           bytes_total_io, bytes_total_io / 1048576.0);

    // -----------------------------------------------------------------------
    // Warmup
    // -----------------------------------------------------------------------
    for (int i = 0; i < cfg.nWarmup; ++i) {
        bench_windowedChEstFilterWrapper<
            float, float, float,
            N_DMRS_GRIDS_PER_PRB,
            N_DMRS_PRB_IN_PER_CLUSTER,
            N_DMRS_INTERP_PRB_OUT_PER_CLUSTER,
            N_DMRS_SYMS>
            <<<grid, block, smemBytes>>>(d_stat, d_dyn);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // -----------------------------------------------------------------------
    // Timed iterations (for nsys / ncu, just run without manual timing)
    // -----------------------------------------------------------------------
    cudaEvent_t evStart, evStop;
    CUDA_CHECK(cudaEventCreate(&evStart));
    CUDA_CHECK(cudaEventCreate(&evStop));

    CUDA_CHECK(cudaEventRecord(evStart));
    for (int i = 0; i < cfg.nIter; ++i) {
        bench_windowedChEstFilterWrapper<
            float, float, float,
            N_DMRS_GRIDS_PER_PRB,
            N_DMRS_PRB_IN_PER_CLUSTER,
            N_DMRS_INTERP_PRB_OUT_PER_CLUSTER,
            N_DMRS_SYMS>
            <<<grid, block, smemBytes>>>(d_stat, d_dyn);
    }
    CUDA_CHECK(cudaEventRecord(evStop));
    CUDA_CHECK(cudaDeviceSynchronize());

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, evStart, evStop));
    const float avg_us = ms / cfg.nIter * 1000.0f;
    printf("  [Timing]\n");
    printf("  Avg kernel time              : %.3f us\n", avg_us);

    // IO-bound roofline estimate
    // RTX 4070 Super: GDDR6X, peak bandwidth ~504 GB/s
    // Adjust BW_GB_S if running on a different GPU.
    constexpr float BW_GB_S = 504.0f;  // RTX 4070 Super peak memory bandwidth (GB/s)
    const float total_io_mb   = (bytes_ls_input + bytes_hest_output) / 1048576.0f;
    const float io_bound_us   = (total_io_mb / 1024.0f) / BW_GB_S * 1e6f; // GB / (GB/s) -> s -> us
    printf("  IO-bound estimate @ %.0f GB/s : %.3f us  (%.1f MB / %.0f GB/s)\n",
           BW_GB_S, io_bound_us, total_io_mb, BW_GB_S);
    printf("  Arithmetic intensity          : %.3f FLOP/byte\n",
           // Each output tone: N_TOTAL_DMRS_GRID_TONES (24) * 2 FMAs = 96 FLOP
           // Total FLOPs ~ nRxAnt * nLayers * nPrb * 12 * 24 * 2
           ((float)cfg.nRxAnt * cfg.nLayers * cfg.nPrb * N_TONES * N_TOTAL_DMRS_GRID_TONES * 2) /
           (bytes_ls_input + bytes_hest_output));

    CUDA_CHECK(cudaEventDestroy(evStart));
    CUDA_CHECK(cudaEventDestroy(evStop));

    // -----------------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------------
    cudaFree(d_stat);
    cudaFree(d_dyn);
    cudaFree(d_ueGrp);
    cudaFree(d_ls);
    cudaFree(d_hest);
    cudaFree(d_coefs4);
    cudaFree(d_coefs);
    cudaFree(d_occIdx);

    printf("  Done.\n\n");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main()
{
    srand(42);

    // Print device info
    int dev;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("Device: %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

    // Two profiling configurations
    BenchConfig configs[] = {
        // name,             nPrb, nRxAnt, nLayers, dmrsMaxLen, warmup, iters
        {"nRxAnt=64 nLayers=2",  273,    64,      2,         1,        5,   100},
        {"nRxAnt=16 nLayers=4",  273,    16,      4,         1,        5,   100},
    };

    for (auto& cfg : configs) {
        run_config(cfg);
    }

    return 0;
}
