#include "graph.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <limits>

#include "graph_cuda.h"

#define BLOCK_SIZE 32

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
    cudaError_t err;

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

    // Allocate device memory for APSP if not already allocated
    if (d_apsp == nullptr) {
        err = cudaMalloc(&d_apsp, n * n * sizeof(float));
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error (malloc): %s\n", cudaGetErrorString(err));
            return;
        }
    }

    err = cudaMemcpy(d_apsp, apsp, n * n * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (memcpy to device): %s\n", cudaGetErrorString(err));
        return;
    }

    // Set up grid and block dimensions
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim((n + block_dim.x - 1) / block_dim.x, 
                  (n + block_dim.y - 1) / block_dim.y);

    // Run Floyd-Warshall algorithm
    for (ui k = 0; k < n; k++) {
        floyd_warshall_kernel<<<grid_dim, block_dim>>>(d_apsp, k, n);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error (kernel): %s\n", cudaGetErrorString(err));
            return;
        }
    }

    // Copy result back to host
    err = cudaMemcpy(apsp, d_apsp, n * n * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (memcpy to host): %s\n", cudaGetErrorString(err));
        return;
    }

    // Check for CUDA errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }
}

__global__ void compute_atd_kernel(float* apsp, ui* neighbors, ept* neighbors_offset, 
                                   float* atd_results, ui n, float alpha) {
    int i = blockIdx.x;
    int j = blockIdx.y;
    
    if (i < n && j < n && i != j) {
        ept start_i = neighbors_offset[i];
        ept end_i = neighbors_offset[i + 1];
        ept start_j = neighbors_offset[j];
        ept end_j = neighbors_offset[j + 1];
        
        ui source_nbr_count = end_i - start_i;
        ui target_nbr_count = end_j - start_j;
        
        float share = (1.0f - alpha) / (source_nbr_count * target_nbr_count);
        float cost_nbr = 0.0f;
        
        for (ept src = start_i; src < end_i; src++) {
            for (ept tgt = start_j; tgt < end_j; tgt++) {
                ui src_node = neighbors[src];
                ui tgt_node = neighbors[tgt];
                cost_nbr += apsp[src_node * n + tgt_node] * share;
            }
        }
        
        float cost_self = alpha * apsp[i * n + j];
        atd_results[i * n + j] = cost_nbr + cost_self;
    }
}