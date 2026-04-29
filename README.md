# CUDA Kernels - From Scratch

A study of GPU kernel optimisation in CUDA, working from a bandwidth-bound vector add up to a tiled matrix multiply that demonstrates the cost-of-memory story behind almost every GPU optimisation. Each kernel is self-contained, instrumented with CUDA events for accurate device-side timing, and validated against a CPU reference.

The point of this repo is not to reinvent cuBLAS - it's to make the optimisation reasoning explicit. What's the bottleneck? How does the optimisation change the bottleneck? What does the profiler tell us that the timing alone doesn't?

## Results

Measured on **NVIDIA GeForce RTX 4070 Laptop GPU**, driver **596.36**, CUDA **12.x**, Windows 11. Numbers are averages over 10 timed iterations after a warm-up launch (3 iterations at 4096³).

| Kernel | Problem size | Time | Throughput |
|---|---|---|---|
| `vector_add` | N = 16M (`1<<24`) floats | 0.99 ms | 204 GB/s |
| `matmul_naive` | 1024 × 1024 × 1024 | 1.92 ms | 1117 GFLOPS |
| `matmul_tiled` | 1024 × 1024 × 1024 | 1.47 ms | 1459 GFLOPS |
| `matmul_naive` | 4096 × 4096 × 4096 | 116.2 ms | 1183 GFLOPS |
| `matmul_tiled` | 4096 × 4096 × 4096 | 84.7 ms | 1622 GFLOPS |

**Tiled matmul achieved a 1.37× speedup at 4096³** through shared-memory reuse, with consistent gains across both problem sizes.

A note on absolute performance: these numbers are below the RTX 4070 Laptop's peak compute throughput. The 4070 Laptop's effective TDP is governed by the host laptop's power profile, and during this benchmarking the GPU was running under a balanced power profile rather than the maximum TGP allowed by the silicon. This means absolute GFLOPS figures here are not directly comparable to desktop or datacentre cards - but the *speedup ratios* between kernels remain valid because both kernels run under identical conditions. This is itself a useful lesson: GPU benchmarks need a controlled hardware environment, which is one reason production GPU work happens on desktop workstations or in the datacentre, not on laptops.

## Build

Requires CUDA Toolkit 12.x, CMake 3.20+, and a C++17 host compiler (MSVC on Windows, GCC/Clang on Linux).

```bash
cmake -S . -B build
cmake --build build --config Release
```

Then run any of the executables:

```bash
# Windows (multi-config generator)
./build/Release/vector_add.exe
./build/Release/matmul_naive.exe
./build/Release/matmul_tiled.exe

# Linux (single-config generator)
./build/vector_add
./build/matmul_naive
./build/matmul_tiled
```

If the build fails with an "unsupported architecture" error, replace `native` in `CMakeLists.txt` with your GPU's SM number (e.g. `86` for Ampere, `89` for Ada Lovelace).

## Kernel 1 - `vector_add.cu`

The CUDA equivalent of "hello world": `c[i] = a[i] + b[i]`, one thread per element.

The result here is *not* the kernel - it's that this kernel will run at roughly the GPU's available DRAM bandwidth and no faster. Three arrays of 16M floats means moving ~192 MB through global memory; the arithmetic is one add per 12 bytes touched, so the bottleneck is memory, not compute. This is the canonical example of a *bandwidth-bound* kernel.

The figure of merit is achieved bandwidth as a fraction of the GPU's peak DRAM bandwidth - and bandwidth-bound kernels behave more predictably than compute-bound ones across thermal/power conditions, which is why this kernel is included as the reference point.

## Kernel 2 - `matmul_naive.cu`

A matrix multiply where each thread computes one output element by reading a full row of A and a full column of B from global memory.

In theory this should be much slower than the tiled version because every output element re-reads K elements of A and K elements of B from DRAM - a factor-of-K redundancy. In practice on Ada Lovelace, the L1/L2 cache hierarchy is sophisticated enough that significant reuse happens automatically: the first thread to read a row pulls it into cache, and subsequent threads reading the same row hit cache rather than DRAM. This is why the naive performance gap on modern hardware is smaller than older textbooks suggest.

The 4096³ run shows the naive kernel holding roughly the same throughput as 1024³, which tells us the working set still partly fits in L2 even at the larger size - or that we're bound by something other than DRAM bandwidth (most likely the FFMA pipeline at the achievable clock).

## Kernel 3 - `matmul_tiled.cu`

The same matmul, restructured around shared memory. Each block of 16×16 threads cooperatively loads a 16×16 tile of A and a 16×16 tile of B into on-chip shared memory, then each thread does 16 multiply-adds against shared memory before sliding the tile.

The arithmetic is unchanged - same FLOPs, same algorithmic I/O - but global-memory traffic drops by a factor of TILE (16×) because each loaded tile is reused by all 16 threads in its row/column, *explicitly* rather than relying on the cache. The result is a measurable speedup at both sizes (1.31× at 1024³, 1.37× at 4096³).

Two implementation details matter:

- **`__syncthreads()` placement.** One barrier after the load (so all threads in the block see the full tile before computing), and one barrier after the compute (so no thread overwrites the tile in the next iteration before others have finished using it).
- **`#pragma unroll` on the inner loop.** Without it, NVCC generates dependent loads from shared memory that limit instruction-level parallelism. With it, the compiler schedules the FFMA chain to keep the multiply-add pipeline saturated.

The story the speedup tells: tiling didn't make the kernel "faster" in some absolute sense - it changed *which resource is the bottleneck*. The naive kernel is partly memory-traffic bound; the tiled kernel pushes that bottleneck towards the SM's compute pipeline.

## What this repo is not

It's not a production GEMM. cuBLAS will beat the tiled kernel here by another 4–10×, primarily through register tiling, vectorised loads (`float4`), warp-level matrix instructions on Tensor Cores, and software pipelining. Those are the techniques used in BLAS libraries; reproducing them properly is a separate project.

This repo deliberately stops at the point where the conceptual lesson - that GPU optimisation is about resource bottleneck shifting, not raw "speed" - is clearest.

## References

- *Programming Massively Parallel Processors* (Kirk & Hwu), chapters on memory hierarchy and tiled matmul.
- NVIDIA *CUDA C++ Programming Guide*, sections on shared memory and the memory hierarchy.
- NVIDIA *CUTLASS* repository for what production-grade tiled GEMM looks like.

## Author

Abhigyan Gandhi - MSc High-Performance Graphics & Games Engineering, University of Leeds (2025).
Graphics and GPU engineer; portfolio at [abhigyan04.github.io](https://abhigyan04.github.io).
