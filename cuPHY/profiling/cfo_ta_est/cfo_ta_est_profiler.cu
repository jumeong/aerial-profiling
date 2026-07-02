#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>
#include <string>

// cuda_fp16.h must come before cuphy.hpp to provide __half22float2
#include <cuda_fp16.h>

// Include necessary cuPHY headers
#include "cuphy.hpp"
#include "cuphy_api.h"
#include "cfo_ta_est.hpp"

// We want to access the kernel function
// The kernel is template-based, so we include the cu file
#include "cfo_ta_est.cu"

#ifndef CUDA_CHECK
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << err << " \"" << cudaGetErrorString(err) << "\"" << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)
#endif

int main(int argc, char** argv) {
    std::string config = "4T4L";
    if (argc > 1) {
        config = argv[1];
    }

    uint32_t nRxAnt = 4;
    uint32_t nLayers = 4;
    if (config == "4T4L") { nRxAnt = 4; nLayers = 4; }
    else if (config == "8T1L") { nRxAnt = 8; nLayers = 1; }
    else if (config == "8T2L") { nRxAnt = 8; nLayers = 2; }
    else {
        std::cerr << "Unsupported configuration: " << config << ". Using 4T4L as fallback." << std::endl;
        nRxAnt = 4;
        nLayers = 4;
    }

    if (nRxAnt > 8) {
        std::cerr << "Error: cfoTaEstLowMimoKernel only supports up to 8 antennas." << std::endl;
        return 1;
    }

    // ----------------------------------------------------------
    // Tensor size calculations
    // ----------------------------------------------------------
    const uint32_t N_PRB = 273;
    // Pad N_PRB to a multiple of N_PRB_PER_THRD_BLK (8) for safe allocation
    constexpr uint32_t N_PRB_PER_THRD_BLK = 8;
    const uint32_t N_PRB_ALLOC = ((N_PRB + N_PRB_PER_THRD_BLK - 1) / N_PRB_PER_THRD_BLK) * N_PRB_PER_THRD_BLK; // 280
    const uint32_t N_TONES = N_PRB_ALLOC * 12;
    const uint32_t N_TIME_CH_EST = 2;
    const uint32_t N_UE_GRP = 1;
    // MAX_ND_SUPPORTED is already defined as 14 in cuphy.h
    const uint32_t MAX_N_UE = 12; // arbitrary max
    const uint32_t N_MAX_LAYERS = 8;

    std::cout << "Profiling cfoTaEstLowMimoKernel configuration: " << config << std::endl;
    std::cout << "nRxAnt=" << nRxAnt << ", nLayers=" << nLayers << ", PRBs=" << N_PRB << std::endl;

    // Allocate GPU buffers
    // tInfoHEst: (N_BS_ANTS, N_LAYERS, NF, NH)
    size_t sz_hEst = nRxAnt * nLayers * N_TONES * N_TIME_CH_EST * sizeof(float2);
    float2* d_hEst; CHECK_CUDA_ERR(cudaMalloc(&d_hEst, sz_hEst));

    // tInfoCfoEst: (MAX_ND_SUPPORTED, MAX_N_UE)
    size_t sz_cfoEst = MAX_ND_SUPPORTED * MAX_N_UE * sizeof(float2);
    float2* d_cfoEst; CHECK_CUDA_ERR(cudaMalloc(&d_cfoEst, sz_cfoEst));

    // tInfoCfoHz: (MAX_N_UE)
    size_t sz_cfoHz = MAX_N_UE * sizeof(float);
    float* d_cfoHz; CHECK_CUDA_ERR(cudaMalloc(&d_cfoHz, sz_cfoHz));

    // tInfoTaEst: (MAX_N_UE)
    size_t sz_taEst = MAX_N_UE * sizeof(float);
    float* d_taEst; CHECK_CUDA_ERR(cudaMalloc(&d_taEst, sz_taEst));

    // tInfoCfoPhaseRot: (MAX_N_TIME_CH_EST, N_MAX_LAYERS, N_MAX_UE_GRPS)
    size_t sz_cfoPhaseRot = 4 * N_MAX_LAYERS * N_UE_GRP * sizeof(float2);
    float2* d_cfoPhaseRot; CHECK_CUDA_ERR(cudaMalloc(&d_cfoPhaseRot, sz_cfoPhaseRot));

    // tInfoTaPhaseRot: (N_MAX_LAYERS, N_MAX_UE_GRPS)
    size_t sz_taPhaseRot = N_MAX_LAYERS * N_UE_GRP * sizeof(float2);
    float2* d_taPhaseRot; CHECK_CUDA_ERR(cudaMalloc(&d_taPhaseRot, sz_taPhaseRot));

    // tInfoCfoTaEstInterCtaSyncCnt
    size_t sz_syncCnt = N_UE_GRP * sizeof(uint32_t);
    uint32_t* d_syncCnt; CHECK_CUDA_ERR(cudaMalloc(&d_syncCnt, sz_syncCnt));
    CHECK_CUDA_ERR(cudaMemset(d_syncCnt, 0, sz_syncCnt));

    // Build cuphyPuschRxUeGrpPrms_t
    cuphyPuschRxUeGrpPrms_t h_ueGrpPrms;
    memset(&h_ueGrpPrms, 0, sizeof(h_ueGrpPrms));

    h_ueGrpPrms.nRxAnt = (uint16_t)nRxAnt;
    h_ueGrpPrms.nLayers = (uint8_t)nLayers;
    h_ueGrpPrms.nPrb = (uint16_t)N_PRB;
    h_ueGrpPrms.startPrb = 0;
    h_ueGrpPrms.enableCfoCorrection = true;
    h_ueGrpPrms.enableToEstimation = true;
    // deltaFHz is not in the struct, likely computed via `mu` (SCS parameter)
    h_ueGrpPrms.mu = 1; // 30kHz SCS
    // cpLength is not in the struct? wait, let me remove cpLength too if it doesn't exist.
    // DMRS info
    h_ueGrpPrms.dmrsSymLoc[0] = 2;
    h_ueGrpPrms.dmrsSymLoc[1] = 11;
    h_ueGrpPrms.dmrsCnt = 2;
    h_ueGrpPrms.ueIdxs[0] = 0; // ueIdx for layer 0
    if(nLayers > 1) h_ueGrpPrms.ueIdxs[1] = 0;
    if(nLayers > 2) h_ueGrpPrms.ueIdxs[2] = 0;
    if(nLayers > 3) h_ueGrpPrms.ueIdxs[3] = 0;

    // tInfoHEst strides (fastest to slowest: N_BS_ANTS, N_LAYERS, NF, NH)
    h_ueGrpPrms.tInfoHEst.pAddr = d_hEst;
    h_ueGrpPrms.tInfoHEst.elemType = CUPHY_C_32F;
    h_ueGrpPrms.tInfoHEst.strides[0] = 1;
    h_ueGrpPrms.tInfoHEst.strides[1] = nRxAnt;
    h_ueGrpPrms.tInfoHEst.strides[2] = nRxAnt * nLayers;
    h_ueGrpPrms.tInfoHEst.strides[3] = nRxAnt * nLayers * N_TONES;

    // tInfoCfoEst strides (MAX_ND_SUPPORTED, MAX_N_UE_PER_UE_GRP)
    h_ueGrpPrms.tInfoCfoEst.pAddr = d_cfoEst;
    h_ueGrpPrms.tInfoCfoEst.elemType = CUPHY_C_32F;
    h_ueGrpPrms.tInfoCfoEst.strides[0] = 1;
    h_ueGrpPrms.tInfoCfoEst.strides[1] = MAX_ND_SUPPORTED;

    h_ueGrpPrms.tInfoCfoHz.pAddr = d_cfoHz;
    h_ueGrpPrms.tInfoCfoHz.elemType = CUPHY_R_32F;
    h_ueGrpPrms.tInfoCfoHz.strides[0] = 1;

    h_ueGrpPrms.tInfoTaEst.pAddr = d_taEst;
    h_ueGrpPrms.tInfoTaEst.elemType = CUPHY_R_32F;
    h_ueGrpPrms.tInfoTaEst.strides[0] = 1;

    h_ueGrpPrms.tInfoCfoPhaseRot.pAddr = d_cfoPhaseRot;
    h_ueGrpPrms.tInfoCfoPhaseRot.elemType = CUPHY_C_32F;
    h_ueGrpPrms.tInfoCfoPhaseRot.strides[0] = 1;
    h_ueGrpPrms.tInfoCfoPhaseRot.strides[1] = 4;
    h_ueGrpPrms.tInfoCfoPhaseRot.strides[2] = 4 * N_MAX_LAYERS;

    h_ueGrpPrms.tInfoTaPhaseRot.pAddr = d_taPhaseRot;
    h_ueGrpPrms.tInfoTaPhaseRot.elemType = CUPHY_C_32F;
    h_ueGrpPrms.tInfoTaPhaseRot.strides[0] = 1;
    h_ueGrpPrms.tInfoTaPhaseRot.strides[1] = N_MAX_LAYERS;

    h_ueGrpPrms.tInfoCfoTaEstInterCtaSyncCnt.pAddr = d_syncCnt;
    h_ueGrpPrms.tInfoCfoTaEstInterCtaSyncCnt.elemType = CUPHY_R_32U;
    h_ueGrpPrms.tInfoCfoTaEstInterCtaSyncCnt.strides[0] = 1;

    cuphyPuschRxUeGrpPrms_t* d_ueGrpPrms;
    CHECK_CUDA_ERR(cudaMalloc(&d_ueGrpPrms, sizeof(cuphyPuschRxUeGrpPrms_t)));
    CHECK_CUDA_ERR(cudaMemcpy(d_ueGrpPrms, &h_ueGrpPrms, sizeof(cuphyPuschRxUeGrpPrms_t), cudaMemcpyHostToDevice));

    puschRxCfoTaEstDynDescr_t dynDescr;
    dynDescr.pDrvdUeGrpPrms = d_ueGrpPrms;
    dynDescr.nUeGrps = N_UE_GRP;
    dynDescr.pFoCompensationBuffers = nullptr;

    // Setup launch geometry
    constexpr uint32_t THRD_GRP_TILE_SIZE = 32;
    uint32_t N_THRD_GRP_TILES_PER_LAYER = (((nRxAnt * 12) + THRD_GRP_TILE_SIZE - 1) / THRD_GRP_TILE_SIZE);
    // N_PRB_PER_THRD_BLK already declared at the top

    uint32_t N_THRDS_PER_LAYER = N_THRD_GRP_TILES_PER_LAYER * THRD_GRP_TILE_SIZE;
    uint32_t nThrdBlksPerUeGrp = (N_PRB + N_PRB_PER_THRD_BLK - 1) / N_PRB_PER_THRD_BLK;

    dim3 gridDim(nThrdBlksPerUeGrp, N_UE_GRP);
    dim3 blockDim(N_THRDS_PER_LAYER, nLayers);

    std::cout << "gridDim(" << gridDim.x << "," << gridDim.y << "," << gridDim.z << ")" << std::endl;
    std::cout << "blockDim(" << blockDim.x << "," << blockDim.y << "," << blockDim.z << ")" << std::endl;

    // Profiling loop
    const int WARMUP_ITERS = 10;
    const int PROFILING_ITERS = 100;
    
    cudaEvent_t start, stop;
    CHECK_CUDA_ERR(cudaEventCreate(&start));
    CHECK_CUDA_ERR(cudaEventCreate(&stop));

    auto launchKernel = [&]() {
        // Zero sync counter before each run
        CHECK_CUDA_ERR(cudaMemset(d_syncCnt, 0, sz_syncCnt));

        // Use appropriate template parameters
        if (nRxAnt == 8 && nLayers == 1) {
            cfo_ta_est::cfoTaEstLowMimoKernel<float2, float2, float, 8, 1, 2, 32, 3, 8><<<gridDim, blockDim>>>(dynDescr);
        } else if (nRxAnt == 8 && nLayers == 2) {
            cfo_ta_est::cfoTaEstLowMimoKernel<float2, float2, float, 8, 2, 2, 32, 3, 8><<<gridDim, blockDim>>>(dynDescr);
        } else if (nRxAnt == 4 && nLayers == 4) {
            cfo_ta_est::cfoTaEstLowMimoKernel<float2, float2, float, 4, 4, 2, 32, 2, 8><<<gridDim, blockDim>>>(dynDescr);
        }
    };

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++) {
        launchKernel();
    }
    CHECK_CUDA_ERR(cudaDeviceSynchronize());

    // Profile
    CHECK_CUDA_ERR(cudaEventRecord(start));
    for (int i = 0; i < PROFILING_ITERS; i++) {
        launchKernel();
    }
    CHECK_CUDA_ERR(cudaEventRecord(stop));
    CHECK_CUDA_ERR(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA_ERR(cudaEventElapsedTime(&ms, start, stop));
    float avg_us = (ms * 1000.0f) / PROFILING_ITERS;

    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Avg Latency: " << std::fixed << std::setprecision(2) << avg_us << " us" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    // Estimate GB10 time (6144 CUDA cores vs 7168 on RTX 4070 Super)
    // ratio = 6144 / 7168 ~ 0.857
    float gb10_us = avg_us * (7168.0f / 6144.0f);
    std::cout << "Estimated GB10 Latency: " << std::fixed << std::setprecision(2) << gb10_us << " us" << std::endl;

    return 0;
}
