# metal-rt-bench

Two small, dependency-free Objective-C programs that measure what Apple's **hardware ray tracing**
actually costs on a given Mac, in milliseconds, with GPU timestamps — no game engine, no shading,
no denoiser. Written because, as of September 2026, there was no published microbenchmark for the
M5 generation (Apple10 GPU family) at all, and the only earlier one
([nelari.us, 2024](https://nelari.us/post/metal-raytracing-performance/), M1–M3 on Sponza) measured
frame times rather than the primitives an engine has to budget for: acceleration-structure build,
refit, and ray throughput per traversal shape.

Everything here was measured on one machine. Results for other Apple GPUs are welcome as pull
requests — `run.sh` writes a file named after the chip into `results/`.

## What is measured

**`rtprobe`** — capability report: GPU families (`Apple6…Apple10`, `Mac2`, `Metal3`, `Metal4`),
the ray-tracing feature flags, MetalFX scaler/denoiser/frame-interpolator support, and
acceleration-structure sizing (AS bytes, build scratch, refit scratch, heap alignment) for
1k–1M-triangle BLASes.

**`rtbench [SEG]`** — a procedural noisy UV sphere of `2·SEG²` triangles (default `512` →
524,288; `1024` → 2,097,152), then:

1. **BLAS build** under each `MTLAccelerationStructureUsage` (PreferFastIntersection,
   PreferFastBuild, MinimizeMemory, Refit, None), plus an in-place **refit** of the Refit-usage AS.
   Median of 12 builds; sizes from `accelerationStructureSizesWithDescriptor:`.
2. **Ray throughput** through the bare PreferFastIntersection BLAS at 1920×1080 (2.07 M rays per
   dispatch, one `dispatchThreads` of 8×8 threadgroups), median of 40 dispatches:
   - closest-hit primary rays from a camera outside the sphere (19 % of pixels hit — the rest
     miss at the root box, so this row is the *cheap* case);
   - the same as any-hit;
   - **interior**: origin at the sphere centre, uniformly random directions, 100 % hit rate, full
     traversal to a leaf on every ray — the load-bearing incoherent-ray number;
   - primary closest-hit + 4 short (0.5 unit) any-hit AO rays per hit pixel.
3. **TLAS arms** — the interior workload again, but traced through an instance acceleration
   structure (what an engine actually binds) with 1 and with 16 instances of the same BLAS, using
   both `intersector<triangle_data, instancing>` and `intersection_query<triangle_data, instancing>`
   — the latter is the form SPIRV-Cross emits for `GL_EXT_ray_query`, i.e. what a Vulkan-style
   engine's ray-query shader becomes on Metal. The query loop counts how often `next()` returns a
   candidate, so the auto-commit behaviour on opaque geometry is measured rather than assumed.

Every workload runs one extra, untimed dispatch into a shared buffer and reports its **measured
hit rate**, so a row can never look fast because it silently missed everything.

Timing is `MTLCommandBuffer.GPUEndTime − GPUStartTime` for a command buffer holding exactly one
encoder, so it includes command-buffer overhead on the GPU timeline but no CPU submission cost.
Nothing is warmed up explicitly; the median over 12/40 runs absorbs the first-dispatch outliers, and
the minimum is printed next to it.

## Requirements

- Apple-silicon Mac with a hardware ray-tracing GPU: **M3 / A17 Pro or later** (`MTLGPUFamilyApple9`+).
  `supportsRaytracing` is YES on every Apple GPU since M1 — it only means *software* RT is
  available; check `supportsFamily(Apple9)` for hardware traversal.
- macOS 26 and Xcode 26/27 (or its Command Line Tools). The source uses `MTLGPUFamilyApple10` and
  `MTLGPUFamilyMetal4`, which exist from the macOS 26 SDK on. No Metal Toolchain is needed — the
  kernel is compiled at run time with `newLibraryWithSource:`.

```sh
./run.sh            # builds both programs, runs rtprobe + rtbench 512 + rtbench 1024,
                    # writes results/<date>-<chip>.txt
./rtbench 256       # or a single size: 2*256*256 = 131,072 triangles
```

## Results — Apple M5 Pro (16-core GPU, 24 GB), 2026-09-01

Raw output: [`results/2026-09-01-Apple-M5-Pro.txt`](results/2026-09-01-Apple-M5-Pro.txt).
macOS 26.5.2 (25F84), Apple clang 21.0.0 (clang-2100.3.23.3), Xcode 27.0 / MacOSX26.5 SDK, on AC
power, machine otherwise idle. `argumentBuffersSupport` = Tier 1, `recommendedMaxWorkingSetSize`
= 17.76 GB, every RT and MetalFX feature flag YES, families Apple6–Apple10, Mac2, Metal3, Metal4.

### BLAS build (median of 12)

| usage | 524,288 tris | 2,097,152 tris |
|---|---|---|
| PreferFastIntersection | **4.771 ms** (109.9 Mtri/s), AS 54.46 MB | **43.447 ms** (48.3 Mtri/s), AS 220.34 MB |
| PreferFastBuild | 3.569 ms (146.9 Mtri/s), AS 54.46 MB | 14.781 ms (141.9 Mtri/s), AS 217.26 MB |
| MinimizeMemory | 3.947 ms (132.8 Mtri/s), AS 50.47 MB | 16.914 ms (124.0 Mtri/s), AS 201.10 MB |
| Refit | 3.589 ms (146.1 Mtri/s), AS 54.73 MB | 14.632 ms (143.3 Mtri/s), AS 218.26 MB |
| **refit** (Refit usage, in place) | **0.656 ms** — 5.5× cheaper than its rebuild, **zero** refit scratch | **2.423 ms** — 6.0× cheaper |
| None | 4.420 ms (118.6 Mtri/s), AS 54.46 MB | 18.080 ms (116.0 Mtri/s), AS 217.26 MB |

AS memory is **~101–110 bytes per triangle** regardless of usage (MinimizeMemory saves 7–8 %);
build scratch is ~57 % of the AS size; refit scratch is 0. Builds scale linearly up to ~0.5 M
triangles; PreferFastIntersection turns **superlinear** (2.80× the time for 4× the triangles) at
2 M while every other usage stays at ~140 Mtri/s — the fast-intersection builder is the one to
keep off the per-frame path for big meshes.

### Ray throughput, 1920×1080 = 2.07 M rays per dispatch (median of 40, min in parentheses)

| workload | 524,288-tri BLAS | 2,097,152-tri BLAS |
|---|---|---|
| closest-hit primary, camera outside, 19 % hit | 0.267 ms (0.266) | 0.279 ms (0.277) |
| any-hit primary, 19 % hit | 0.276 ms (0.275) | 0.348 ms (0.340) |
| **interior, 100 % hit, incoherent** | **1.028 ms (1.024) → 2.02 Grays/s** | **1.749 ms (1.716) → 1.19 Grays/s** |
| primary + 4 AO any-hit rays (13 % of px hit, 0.19 occluders/px) | 0.571 ms (0.570) | 0.663 ms (0.660) |

The primary rows are dominated by the 81 % of rays that miss the root box; read them as the floor
of dispatch + one box test (~7.5 Grays/s), not as traversal throughput. The **interior row is the
number to budget with**: one fully incoherent, always-hitting ray per 1080p pixel costs **~1.0 ms
against 0.5 M triangles and ~1.75 ms against 2 M** on this GPU.

### TLAS arms — the interior workload through an instance acceleration structure

| | 524,288-tri BLAS | 2,097,152-tri BLAS |
|---|---|---|
| bare BLAS, `intersector` (from the table above) | 1.028 ms | 1.749 ms |
| TLAS ×1, `intersector<triangle_data, instancing>` | **1.390 ms** (+35 %) | **1.826 ms** (+4 %) |
| TLAS ×1, `intersection_query<triangle_data, instancing>` | **1.681 ms** (+21 % over intersector) | **2.062 ms** (+13 %) |
| TLAS ×16, `intersector` | 1.391 ms | 1.834 ms |
| TLAS ×16, `intersection_query` | 1.681 ms | 2.064 ms |
| TLAS build, 1 / 16 instances | 0.028 / 0.058 ms (0.5 / 3.7 KB) | 0.027 / 0.057 ms |

Observations:

- **The instance level costs a fixed ~0.3–0.4 ms per 2 M rays** on top of BLAS traversal
  (+35 % on the small mesh, +4 % on the large one, where leaf traversal dominates). Sixteen
  instances cost the same as one here because the fifteen extra instances sit outside the shell,
  behind every ray's first hit — this measures the TLAS *level*, not TLAS *occupancy*.
- **`intersection_query` is 13–21 % slower than `intersector` on identical work**, and `next()`
  handed back **0.000 candidates per ray**: on opaque geometry the query auto-commits the closest
  triangle exactly like Vulkan's `rayQueryProceedEXT`, so the gap is pure API/codegen overhead of
  the resumable query object, not extra traversal. This is the tax a Vulkan-style engine pays for
  routing `GL_EXT_ray_query` through SPIRV-Cross instead of writing `intersector` MSL by hand.
- All rows report a measured 100 % hit rate.

## Caveats

- One machine, one session, one thermal state; these are medians of short bursts on an idle
  desktop-class chip on AC power, not sustained numbers. A previous run of the same 2 M-triangle
  interior workload on the same machine measured 2.04 ms vs 1.75 ms here — expect ±15 % between
  sessions on the large-mesh rows; the 0.5 M rows reproduced within 1 %.
- One BLAS shape (a bumpy sphere: uniform triangle size, no thin/long triangles, no overlap), so
  build rates for real meshes will differ; the sphere is the same for every row, which is what
  makes the rows comparable.
- Ray counts assume one ray per thread and no shading; there is no material evaluation, texture
  fetch, or denoising, so an engine's "RT pass" will cost more than the traversal number alone.
- `GPUEndTime − GPUStartTime` includes the command buffer's own overhead on the GPU timeline
  (visible as the ~0.27 ms floor on the primary rows).
- MetalFX support is a `supportsDevice:` query, not a measurement.

## Files

| file | |
|---|---|
| `rtprobe.m` | capability + AS sizing probe (needs `-framework MetalFX`) |
| `rtbench.m` | the benchmark host: mesh, BLAS builds, refit, workloads, TLAS arms, validation readback |
| `rt.metal` | the two compute kernels: `trace` (bare BLAS, 4 modes) and `trace_tlas` (instancing, `intersector` vs `intersection_query`) |
| `run.sh` | build + run + save to `results/` |
| `results/` | one raw output file per machine and date |

## License

MIT — see `LICENSE`.
