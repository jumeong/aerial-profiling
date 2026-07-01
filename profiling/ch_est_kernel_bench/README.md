# windowedChEstFilterNoDftSOfdmKernel Standalone Profiling Bench

단독으로 `windowedChEstFilterNoDftSOfdmKernel`을 launch하여 커널 실행 시간을 측정하고,  
필요 시 Nsight Systems / Nsight Compute로 상세 분석하는 standalone 환경입니다.

## 파일 구성

```
bench_windowed_chest.cu   ← 커널 + wrapper + 드라이버 (all-in-one)
CMakeLists.txt            ← standalone cmake 프로젝트 (sm_89 기본값)
README.md                 ← 이 파일
```

## 타겟 설정

| Config | nPrb | nRxAnt | nLayers | dmrsMaxLen | blockDim | gridDim | smem |
|--------|------|--------|---------|-----------|----------|---------|------|
| A | 273 | 64 | 2 | 1 | (48, 4, 1) | (137, 16, 1) | 3072 B |
| B | 273 | 16 | 4 | 1 | (96, 4, 1) | (137,  4, 1) | 6144 B |

- `273 % 4 = 1` → `N_DMRS_PRB_IN=4, N_INTERP_PRB_OUT=2` (원본 dispatch 로직과 동일)
- `dmrsMaxLen=1` → `N_DMRS_SYMS=1`
- `nRxAntPerBlock=4` (production 휴리스틱 적용 결과)

## 빌드

GPU: **RTX 4070 Super (Ada Lovelace, sm_89)**

```bash
cmake -B build   # 기본값 sm_89 자동 적용
cmake --build build -j$(nproc)
```

다른 GPU를 사용할 경우:

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=<arch>
# sm_80=A100, sm_86=RTX30xx, sm_89=RTX40xx/Ada, sm_90=H100
```

## 실행 (커널 시간 및 IO 크기 자동 출력)

```bash
./build/bench_windowed_chest
```

**ncu 없이도** `cudaEventRecord`로 측정한 평균 커널 실행 시간 및 이론적인 IO 병목 한계(Roofline)가 바로 출력됩니다.

```text
Device: NVIDIA GeForce RTX 4070 SUPER (SM 8.9)

=== Config: nRxAnt=64 nLayers=2 ===
  ...
  [IO Sizes]
  Input  LS estimates  (tInfoDmrsLSEst)   :    1677312 bytes  = 1.600 MB
  Input  Interp coefs  (tPrmFreqInterpCoefs4, cached):   7200 bytes  = 7.031 KB
  Output H estimates   (tInfoHEst)         :    3354624 bytes  = 3.200 MB
  Total IO (excl. cached coefs)            :    5031936 bytes  = 4.800 MB

  [Timing]
  Avg kernel time              : XX.XXX us
  IO-bound estimate @ 504 GB/s : 9.470 us  (4.8 MB / 504 GB/s)
  Arithmetic intensity          : 0.576 FLOP/byte
  Done.
```

### ⏱️ 시간 측정 원리 (`cudaEvent` vs `ncu`)

- **`cudaEventElapsedTime` (현재 방식)**: GPU 스트림에 시작/종료 마커를 삽입하여 측정합니다. 커널이 끝난 직후의 타임스탬프를 GPU가 기록하므로 **CPU 대기 시간(Sync Overhead)이나 launch 지연 없이 순수 커널 실행 시간(Wall Time)**을 정확히 측정합니다. "이 커널의 레이턴시가 얼마인가?"를 확인할 때 사용합니다.
- **`ncu` (Nsight Compute)**: 하드웨어 카운터를 읽기 위해 커널을 여러 번 재실행(Replay)합니다. 이 과정에서 발생하는 오버헤드 때문에 **실행 시간이 실제보다 훨씬 길게 측정**됩니다. 시간 측정이 아닌 "왜 느린가?"(L2 Hit rate, SMEM 대역폭 등 병목 분석)를 파악할 때 사용합니다.

### 💾 IO 대역폭 및 병목 분석

- **504 GB/s의 출처**: RTX 4070 Super의 공식 스펙입니다. (GDDR6X, 192-bit bus, 21 Gbps/pin → 192/8 * 21 = 504 GB/s).
- 커널 실행 전 `cudaMemcpy`는 테스트용 더미 데이터 셋업 과정이며, **커널 측정 시간에는 포함되지 않습니다**. 실제 파이프라인에서는 이전 커널이 이미 VRAM에 올려둔 데이터를 읽으므로 PCIe 전송이 생략됩니다.
- 측정한 커널 시간이 **IO-bound estimate**와 비슷하다면 완벽하게 메모리 대역폭을 최대로 활용하고 있는 것입니다. 시간이 훨씬 길다면 연산/레지스터 등 다른 병목이 존재함을 뜻합니다.

---

## 🔍 상세 프로파일링 (선택)

커널 시간만으로 부족하고 **왜 느린지** 분석하려면 Nsight 툴을 사용합니다.

### Nsight Systems — 타임라인 / 전체 흐름

```bash
nsys profile --trace=cuda --output=chest_nsys ./build/bench_windowed_chest
nsys-ui chest_nsys.nsys-rep
```

### Nsight Compute — 커널 내부 상세 분석

```bash
# 전체 metric set (occupancy, memory BW, warp stall 등)
ncu --set full -o chest_ncu ./build/bench_windowed_chest
```

---

## 💡 설계 및 분석 노트

### Tensor Core 미사용 확인
전체 프로젝트 코드를 분석한 결과, **cuPHY (`ch_est`, `channel_eq` 등) 계층에서는 Tensor Core를 전혀 사용하지 않습니다**.
- `mma.h` include나 `wmma` 주석 흔적은 있지만, 실제 동작하는 코드는 존재하지 않습니다.
- 현재 측정하는 `windowedChEstFilterNoDftSOfdmKernel` 역시 Tensor Core와 무관한 **순수 CUDA FP32 FMA(Multiply-Accumulate) 연산**으로 구성되어 있습니다.
- (참고로 Tensor Core는 `cuMAC/examples/tensorCoreGEMM` 데모 예제에서만 사용됩니다.)

### TStorage / TCompute 타입 (FP16 vs FP32)
실제 운영(Production) 코드와 벤치마크 코드 간의 핵심 차이점입니다:
- **Production (`ch_est.cu`)**: 내부 연산(`TCompute`)은 항상 `float`로 강제되지만, 입출력 버퍼(`TStorage`, `TDataRx`)는 런타임 설정에 따라 대역폭 절약을 위해 주로 **`__half` (FP16)**를 사용합니다.
- **Bench 환경**: 복잡한 의존성(cuPHY 타입 캐스팅 매크로 등)을 끊어내고 독립적으로 빌드하기 위해, **연산과 입출력 버퍼를 모두 `float`로 고정**했습니다.
  
> ⚠️ **영향**: 벤치마크는 모든 데이터가 8바이트(`float2`)이므로, **Production(FP16 사용 시) 대비 메모리 IO 량이 정확히 2배 높습니다**. 만약 `ncu` 분석 시 심한 Memory-Bound로 판별된다면, 이 타입 차이 때문일 수 있습니다. (연산량(ALU) 자체는 프로덕션과 100% 동일합니다.)

### `__device__` 함수를 단독 launch하는 방법
원본 `windowedChEstFilterNoDftSOfdmKernel`은 `static __device__ void`이므로 직접 `<<<>>>` 로 launch 불가합니다.
이 bench는 **Thin Wrapper Kernel** 패턴을 사용합니다:

```
bench_windowedChEstFilterWrapper (global kernel)
  ├─ shared memory 초기화 (sh_shiftSeq=1+j0, sh_unShiftSeq=1+j0)
  └─ windowedChEstFilterNoDftSOfdmKernel (device func, 원본 로직 그대로)
```
`sh_shiftSeq / sh_unShiftSeq`를 identity(`1+j0`)로 초기화하면 delay_mean=0과 동일하므로 커널 자체 연산만 순수하게 측정 가능합니다.
