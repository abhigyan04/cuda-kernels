#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <chrono>

// CUDA error-check macro. Wrap every CUDA API call in this for sanity.
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess) {                                                \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n",                       \
                         __FILE__, __LINE__, cudaGetErrorString(err));           \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

// __global__ marks a function callable from host, executed on device.
// Each thread computes one element of c[i] = a[i] + b[i].
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int N = 1 << 24;            // 16M elements
    const size_t bytes = N * sizeof(float);

    // Allocate and initialise host memory
    std::vector<float> h_a(N, 1.0f);
    std::vector<float> h_b(N, 2.0f);
    std::vector<float> h_c(N, 0.0f);

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    // Configure launch: 256 threads per block, enough blocks to cover N.
    const int threads_per_block = 256;
    const int blocks = (N + threads_per_block - 1) / threads_per_block;

    // Time the kernel using CUDA events (more accurate than CPU clock for GPU work)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    vector_add<<<blocks, threads_per_block>>>(d_a, d_b, d_c, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    // Copy result back and verify
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < N; ++i) {
        if (h_c[i] != 3.0f) { ok = false; break; }
    }

    // Bandwidth: 3 arrays touched (2 read, 1 write) * bytes / time
    double gb = 3.0 * bytes / 1e9;
    double bw = gb / (ms / 1000.0);

    std::printf("Vector add: N = %d, time = %.3f ms, bandwidth = %.2f GB/s, %s\n",
                N, ms, bw, ok ? "PASS" : "FAIL");

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
