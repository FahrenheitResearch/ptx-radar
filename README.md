# ptx-radar

Hand-written PTX speedups on top of the [cursdar3](https://github.com/FahrenheitResearch/cursdar3)
NEXRAD radar interrogation workstation. Every `__global__` in the codebase has a
hand-controlled PTX twin loaded at runtime via the CUDA Driver API. The
nvcc-compiled kernels remain as transparent fallbacks, so the build is never
worse than upstream and a single environment variable flips the entire app back
to the original codegen path for A/B testing.

The radar engine, the ingest pipeline, the workstation shell, and the ImGui UI
all come from upstream cursdar3 unchanged. This repo replaces the GPU compute
substrate underneath them.

## What changed vs upstream cursdar3

### 1. Native Blackwell SASS

Upstream's `CMAKE_CUDA_ARCHITECTURES native` was silently picking `sm_75`
(Turing) on RTX 5090 boxes. Forced to:

```cmake
set(CMAKE_CUDA_ARCHITECTURES "86;89;120;120-virtual" CACHE STRING ... FORCE)
```

so the binary now ships real `sm_86` / `sm_89` / `sm_120` SASS plus a
`compute_120` PTX fallback. This is a one-line change but probably the largest
single performance contributor in the diff — every kernel runs native Blackwell
code instead of JIT'd Turing PTX.

### 2. Hand-PTX kernel module

`src/cuda/ultra_kernels.ptx` is **316 KB / 10,193 lines** containing **33
hand-written PTX entry kernels** — every `__global__` in the upstream tree.
The PTX is embedded into the binary at build time via `configure_file` and
loaded once at startup with `cuModuleLoadData`. Each kernel is exposed as a
`CUfunction` handle and dispatched via `cuLaunchKernel` from a thin wrapper at
the original CUDA call site, with a transparent fallback to the nvcc kernel if
the handle is null or the launch errors:

```cpp
if (ultra_ptx::tryLaunch(ultra_ptx::k_singleStationKernel,
                         gx, gy, 1, bx, by, 1,
                         shared, stream, args, "singleStationKernel")) {
    return;  // hand-PTX path took it
}
nvccKernel<<<grid, block, shared, stream>>>(args...);  // fallback
```

#### Coverage matrix

| File | Kernels (all hand-PTX) |
|---|---|
| `renderer.cu` | `clearKernel`, `clearKernel_v4`, `forwardResolveKernel`, `singleStationKernel`, `nativeRenderKernel`, `forwardRenderKernel`, `initGridCellsKernel`, `buildGridKernel` |
| `preprocess.cu` | `dealiasVelocityKernel`, `ringStatsKernel`, `zeroSuppressedGatesKernel` |
| `gpu_tensor.cu` | `generateUniformAzimuthsKernel`, `buildRadialMapKernel`, `buildGateMapKernel`, `buildTensorKernel` |
| `gpu_detection.cu` | `tdsCandidateKernel`, `hailCandidateKernel`, `mesoCandidateKernel`, `supportExtremumKernel`, `compactCandidatesKernel` |
| `gpu_pipeline.cu` | `parseMsg31Kernel`, `findMessageOffsetsKernel`, `findVariableOffsetsKernel`, `scanSweepMetadataKernel`, `collectLowestSweepIndicesKernel`, `collectSweepIndicesKernel`, `selectProductMetaAllKernel`, `transposeKernel`, `extractAzimuthsKernel` |
| `volume3d.cu` | `buildVolumeKernel`, `smoothVolumeKernel`, `crossSectionKernel`, `rayMarchKernel` (2,527 lines on its own) |

### 3. Hand-applied micro-optimizations (ULTRA-OPTs)

The hand-PTX is mostly extracted from nvcc output (renamed and label-rewritten
so symbols don't collide) plus a small number of targeted instruction-level
folds the compiler doesn't do on its own:

- **ULTRA-OPT #1** — `lat * pi / 180` mul+mov+div+cos collapsed to mul+cos:
  ```ptx
  // before: 3 ops, ~17 cycle critical path
  mul.f32  %fA, lat, 0f40490FDB    // * pi
  mov.f32  %fB,      0f43340000    // 180.0
  div.approx.f32 %fC, %fA, %fB     // / 180     <-- ~14 cyc div
  cos.approx.f32 %fD, %fC

  // after: 1 op, ~3 cycle path
  mul.f32  %fC, lat, 0f3C8EFA35    // * (pi/180)
  cos.approx.f32 %fD, %fC
  ```
  Hot in `nativeRenderKernel` (per station, per pixel) and `singleStationKernel`.

- **ULTRA-OPT #2** — `(norm * 254 + 1) / 128` colormap-coord fma+mov+div
  collapsed into a single fma. Both constants (`254/128 = 1.984375` and
  `1/128 = 0.0078125`) are exact in IEEE-754 single, so the fold is
  bit-identical to the source at the endpoints.
  ```ptx
  // before
  fma.f32 %fA, norm, 0f437E0000, 0f3F800000   // *254 + 1
  mov.f32 %fB,                   0f43800000   // 128
  div.approx.f32 %fC, %fA, %fB                // / 128

  // after
  fma.f32 %fC, norm, 0f3FFE0000, 0f3C000000   // norm*1.984375 + 0.0078125
  ```

- **ULTRA-OPT #3 / #4** — power-of-two divides by 256 / 180 in the volume
  kernels replaced with multiplies by precomputed reciprocals.

- **`ultra_clearKernel_v4`** — hand-written from scratch (not extracted from
  nvcc): vectorized framebuffer clear that writes 4 RGBA pixels per thread via
  a single `st.global.v4.b32`, with a scalar tail for partial bundles at the
  right edge. Used by `clearOutputBuffer` when present.

- **Parallel `selectProductMetaAllKernel`** — replaces upstream's
  `<<<1, 1>>>` serial scan launched `NUM_PRODUCTS` times in a row with a
  single `<<<NUM_PRODUCTS, 256>>>` launch that does a per-block parallel
  reduction in shared memory. (Lives in nvcc fallback now after a regression
  caused the parallel version to ship in PTX only — see the kill-switch
  section below.)

#### Measured register pressure (`cuobjdump --dump-resource-usage`)

| Kernel | nvcc REG | hand-PTX REG | Δ |
|---|---|---|---|
| `nativeRenderKernel` | 42 | **34** | −19 % (~25 % higher Blackwell occupancy) |
| `singleStationKernel` | 31 | **28** | −10 % |
| `buildVolumeKernel` | 45 | **40** | −11 % |
| `forwardRenderKernel` | 37 | 37 | 0 (verbatim — see below) |
| `rayMarchKernel` | 48 | 48 | 0 |

Zero register spills, zero stack usage, zero local memory across **all 33**
kernels.

### 4. Non-PTX optimizations applied along the way

These are CUDA C / host-side changes that ship alongside the PTX module and
benefit the nvcc fallback path too:

- **Per-frame H↔D copy elimination.** `renderNative` (the show-all multi-station
  compositor) was copying ~1 MB of `SpatialGrid` plus the per-station info
  table from host to device every frame. New dirty flags
  (`s_gridDirty`, `s_stationsDirty`) skip the upload when nothing changed.
- **`buildSpatialGridGpu`** writes directly into the persistent
  `d_spatialGrid` device buffer instead of round-tripping through host memory.
- **Removed the per-pane `cudaDeviceSynchronize`** in `App::render`. The kernel
  launches and the panel-texture `cudaMemcpy2DToArray` already serialize on
  the default stream, so the sync was a pure CPU stall — 4 stalls per frame
  in 4-pane mode.
- **`__launch_bounds__`** added to every hot kernel.
- **`-Xptxas=-v`** kept on for visibility.

### 5. UI / quality-of-life

- **Global UI zoom hotkeys** (`=` / `-` / `0` for 60 %–300 %, `Shift+=` /
  `Shift+-` / `Shift+0` for radar render-scale "chunky pixels" mode) for
  phone-remote sessions where the desktop client can't pinch-zoom the host.
- A small fade-in HUD shows the current zoom percentage on every adjust.

## Building

Same as upstream cursdar3 — Windows + Visual Studio Build Tools 2022 + CUDA
Toolkit 13.0 (or compatible) + Ninja:

```
build.bat
```

Drops `build/ptx-radar.exe`. Builds GLFW, Dear ImGui, nlohmann_json, and
bzip2 from source via FetchContent on first run.

## Running

```powershell
.\build\ptx-radar.exe
```

On startup, the stderr log line confirms the PTX module loaded:

```
[ultra-ptx] loaded 33 hand-written kernels from embedded PTX (316120 bytes of source)
```

## Kill switch (regression escape hatch)

Two environment variables can disable hand-PTX kernels at startup, falling
back to the nvcc-compiled originals. Useful for A/B testing and for the rare
case where a hand-PTX kernel produces a bad result you didn't catch.

```powershell
# Disable ALL hand-PTX kernels — every launch falls back to nvcc
$env:CURSDAR3_NO_PTX = "1"
.\build\ptx-radar.exe

# Disable a specific subset (comma-separated substring match against
# the dbg_name passed to ultra_ptx::tryLaunch)
$env:CURSDAR3_NO_PTX_KERNELS = "forwardRender,singleStation"
.\build\ptx-radar.exe
```

When the master kill switch is active you'll see:

```
[ultra-ptx] CURSDAR3_NO_PTX set: all hand-PTX kernels disabled, using nvcc fallbacks only
```

To clear in PowerShell:

```powershell
Remove-Item env:CURSDAR3_NO_PTX -ErrorAction SilentlyContinue
Remove-Item env:CURSDAR3_NO_PTX_KERNELS -ErrorAction SilentlyContinue
```

## Known issues

- **`forwardRenderKernel` ships verbatim from nvcc.** The two ULTRA-OPTs
  applied cleanly to `singleStationKernel` and `nativeRenderKernel` but
  introduced a subtle reflectivity-colormap regression when applied to
  `forwardRenderKernel` (storms read as all-purple because low-dBZ gates went
  missing). The hand-PTX version of this one kernel therefore ships with
  renames-only, no instruction-level edits. SASS register count is identical
  to nvcc. The two opts could probably be re-applied with more care — left as
  future work.

- **Thrust merge-sort kernels are still nvcc.** `gpu_pipeline.cu` calls
  `thrust::sort` which expands to several thousand-line cub merge-sort
  kernels. Those are auto-generated by thrust at compile time and are not
  directly launched by user code, so they're not in the hand-PTX module.

## Honest perf disclaimer

The headline number on this repo is "33 hand-PTX kernels", but the largest
*measured* speedup vs upstream comes from the Blackwell SASS arch fix, the
dirty-flag elimination, and the per-pane sync removal — none of which are
PTX-specific. The hand-PTX work itself is mostly verbatim nvcc output with a
few targeted instruction folds; the register-pressure wins on the three
biggest kernels are real but the resulting frame-time delta on a 5090 is
unmeasured. If you want a number, the easiest way is to launch with and
without `CURSDAR3_NO_PTX=1` and compare the FPS counter in the title bar, or
profile under `nsys`.

## License

MIT. Inherited from the upstream
[cursdar3](https://github.com/FahrenheitResearch/cursdar3) repo. The original
copyright notice is preserved in `LICENSE`.

## Upstream

Tracks [`FahrenheitResearch/cursdar3`](https://github.com/FahrenheitResearch/cursdar3)
— the radar engine, ingest pipeline, workstation shell, and ImGui UI are all
unchanged from upstream. See `git log` for the full history.
