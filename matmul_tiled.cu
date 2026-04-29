#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess) {                                                \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n",                       \
                         __FILE__, __LINE__, cudaGetErrorString(err));           \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

constexpr int TILE = 16;

// Tiled matmul. Each block computes a TILE x TILE output sub-block of C.
// Threads cooperatively load TILE x TILE tiles of A and B into shared mem,
// then iterate over K in tile-sized chunks.
__global__ void matmul_tiled(const float* A, const float* B, float* C,
                             int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;

    // Slide a TILE-wide window across the K dimension.
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        // Cooperative load: each thread loads one element of A's tile and one of B's.
        int a_col = t * TILE + tx;
        int b_row = t * TILE + ty;

        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();   // wait for all threads in block to finish loading

        // Each thread does TILE multiplies-and-adds against shared memory.
        // No bank conflicts because of the TILE alignment + float access pattern.
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();   // wait before overwriting tiles in next iteration
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

int main() {
    const int M = 1024, N = 1024, K = 1024;
    const size_t bytes_A = M * K * sizeof(float);
    const size_t bytes_B = K * N * sizeof(float);
    const size_t bytes_C = M * N * sizeof(float);

    std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N), h_C_ref(M * N);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : h_A) v = dist(rng);
    for (auto& v : h_B) v = dist(rng);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    matmul_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);  // warm-up

    const int iters = 10;
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        matmul_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    float ms_per = ms / iters;

    double flops = 2.0 * M * N * K;
    double gflops = flops / (ms_per / 1000.0) / 1e9;

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes_C, cudaMemcpyDeviceToHost));

    // Spot-check a few elements against a CPU reference (slow but correct)
    bool ok = true;
    for (int i = 0; i < 5; ++i) {
        int row = (i * 211) % M;
        int col = (i * 307) % N;
        float ref = 0.0f;
        for (int k = 0; k < K; ++k) ref += h_A[row * K + k] * h_B[k * N + col];
        if (std::abs(ref - h_C[row * N + col]) > 1e-2f) { ok = false; break; }
    }

    std::printf("Matmul tiled: %dx%dx%d, time = %.3f ms, performance = %.1f GFLOPS, %s\n",
                M, N, K, ms_per, gflops, ok ? "PASS" : "FAIL");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
