#include "graph.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <limits>

__global__ void floyd_warshall_kernel(float* dist, int k, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n && j < n) {
        int ij = i * n + j;
        int ik = i * n + k;
        int kj = k * n + j;
        float alt = dist[ik] + dist[kj];
        if (alt < dist[ij]) {
            dist[ij] = alt;
        }
    }
}

void Graph::computeAPSP() {
    // Allocate host memory for APSP
    apsp = new float[n * n];
    
    // Initialize APSP matrix
    for (ui i = 0; i < n; i++) {
        for (ui j = 0; j < n; j++) {
            apsp[i * n + j] = (i == j) ? 0.0f : std::numeric_limits<float>::infinity();
        }
    }

    // Set initial distances based on graph structure and weights
    for (ui i = 0; i < n; i++) {
        ept start = neighbors_offset[i];
        ept end = neighbors_offset[i + 1];
        for (ept j = start; j < end; j++) {
            ui neighbor = neighbors[j];
            apsp[i * n + neighbor] = weights[j];
        }
    }

    // Allocate device memory for APSP
    cudaMalloc(&d_apsp, n * n * sizeof(float));
    cudaMemcpy(d_apsp, apsp, n * n * sizeof(float), cudaMemcpyHostToDevice);

    // Set up grid and block dimensions
    dim3 block_dim(32, 32);
    dim3 grid_dim((n + block_dim.x - 1) / block_dim.x, 
                  (n + block_dim.y - 1) / block_dim.y);

    // Run Floyd-Warshall algorithm
    for (ui k = 0; k < n; k++) {
        floyd_warshall_kernel<<<grid_dim, block_dim>>>(d_apsp, k, n);
        cudaDeviceSynchronize();
    }

    // Copy result back to host
    cudaMemcpy(apsp, d_apsp, n * n * sizeof(float), cudaMemcpyDeviceToHost);

   

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }
}