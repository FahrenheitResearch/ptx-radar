// Shared access to the hand-written PTX kernels loaded by renderer.cu.
//
// renderer.cu owns the CUmodule and runs cuModuleLoadData / GetFunction at
// gpu::init(). This header lets the other .cu files (preprocess, gpu_tensor,
// gpu_detection, ...) reach the function handles so they can launch the
// hand-PTX kernels via cuLaunchKernel, with transparent fallback to the
// nvcc-compiled kernel if the PTX handle is null.
//
// All handles default to nullptr until gpu::init() runs. Always check before
// launching:
//
//     if (ultra_ptx::k_dealiasVelocityKernel) { cuLaunchKernel(...); ... }
//     else { /* fallback to nvcc kernel */ }
//
#pragma once

#include <cuda.h>
#include <cstdio>

namespace ultra_ptx {

// Render path (Round 1-3): owned + initialized in renderer.cu but kept as
// file-static there. Not exposed via this header because the call sites are
// already inside renderer.cu.

// Round 4 batch: data ingest, tensor reinterpolation, detection.
extern CUfunction k_dealiasVelocityKernel;
extern CUfunction k_ringStatsKernel;
extern CUfunction k_zeroSuppressedGatesKernel;

extern CUfunction k_generateUniformAzimuthsKernel;
extern CUfunction k_buildRadialMapKernel;
extern CUfunction k_buildGateMapKernel;
extern CUfunction k_buildTensorKernel;

extern CUfunction k_tdsCandidateKernel;
extern CUfunction k_hailCandidateKernel;
extern CUfunction k_mesoCandidateKernel;
extern CUfunction k_supportExtremumKernel;
extern CUfunction k_compactCandidatesKernel;

// Round 5: 3D volume + cross-section.
extern CUfunction k_buildVolumeKernel;
extern CUfunction k_smoothVolumeKernel;
extern CUfunction k_crossSectionKernel;

// Round 6: ray-march for 3D mode.
extern CUfunction k_rayMarchKernel;

// Round 7: pipeline parsing + spatial grid construction.
extern CUfunction k_parseMsg31Kernel;
extern CUfunction k_findMessageOffsetsKernel;
extern CUfunction k_findVariableOffsetsKernel;
extern CUfunction k_scanSweepMetadataKernel;
extern CUfunction k_collectLowestSweepIndicesKernel;
extern CUfunction k_collectSweepIndicesKernel;
extern CUfunction k_selectProductMetaAllKernel;
extern CUfunction k_transposeKernel;
extern CUfunction k_extractAzimuthsKernel;
extern CUfunction k_initGridCellsKernel;
extern CUfunction k_buildGridKernel;

// Mirrored constant-memory globals from volume3d.cu and renderer.cu.
// renderer.cu populates these at module-load time via cuModuleGetGlobal,
// and exposes upload helpers so volume3d.cu can keep them in sync with the
// CUDA-C __constant__ versions whenever cudaMemcpyToSymbol runs.
//
// All three are zero-byte until the PTX module is loaded; the helpers below
// no-op safely if PTX never came up.
void uploadConstSweeps(const void* host_ptr, size_t bytes);
void uploadConstNumSweeps(int value);
void uploadConstColorTable(const void* host_ptr, size_t bytes);

// Convenience: stringified CUDA driver result code.
const char* err_str(CUresult res);

// Per-kernel launch counters for the runtime self-check. The shutdown
// printout reads these so you can confirm every loaded PTX kernel actually
// got called during normal operation, not just loaded.
void noteLaunch(const char* name);
void dumpLaunchCounts();

// Global kill switch. When true, tryLaunch always returns false so every
// call site falls back to the nvcc-compiled kernel. Set at gpu::init() from
// the CURSDAR3_NO_PTX env var. Useful for A/B comparison to isolate which
// kernel (if any) has a semantic regression.
extern bool g_disablePtx;

// Per-kernel disable list. Set via CURSDAR3_NO_PTX_KERNELS=nativeRender,forwardRender
// env var — comma-separated substrings matched against the dbg_name.
bool isKernelDisabled(const char* dbg_name);

// Compact try-launcher: returns true on successful PTX launch, false if the
// handle is null or cuLaunchKernel reported an error (and prints to stderr).
// Caller falls back to the nvcc kernel on false.
//
// Defined inline so it gets one copy per TU but no inter-TU dependency.
inline bool tryLaunch(CUfunction fn,
                      unsigned gx, unsigned gy, unsigned gz,
                      unsigned bx, unsigned by, unsigned bz,
                      unsigned shared_bytes,
                      CUstream stream,
                      void** args,
                      const char* dbg_name) {
    if (g_disablePtx || !fn || isKernelDisabled(dbg_name)) return false;
    CUresult res = cuLaunchKernel(fn, gx, gy, gz, bx, by, bz,
                                  shared_bytes, stream, args, nullptr);
    if (res == CUDA_SUCCESS) {
        noteLaunch(dbg_name);
        return true;
    }
    fprintf(stderr, "[ultra-ptx] %s launch failed: %s; falling back to nvcc\n",
            dbg_name, err_str(res));
    return false;
}

}  // namespace ultra_ptx
