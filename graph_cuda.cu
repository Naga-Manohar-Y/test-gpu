#include "graph.h"
// #include <cuda_runtime.h>

__global__ void computeATDandRicciKernel(unsigned int n, unsigned int* degree, ept* neighbors_offset, 
                                         unsigned int* neighbors, float* atd_results, float* ricci_results, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        ept start = neighbors_offset[idx];
        ept end = neighbors_offset[idx + 1];
        float atd_sum = 0.0f;

        for (ept j = start; j < end; j++) {
            unsigned int neighbor = neighbors[j];
            // Simplified ATD calculation
            float avg_degree = (degree[idx] + degree[neighbor]) / 2.0f;
            float ricci = 1.0f - (avg_degree / 1.0f);  // assuming weight = 1 for simplification
            ricci_results[j] = ricci;
            atd_sum += ricci;
        }
        atd_results[idx] = atd_sum / (end - start);
    }
}

void Graph::computeRicciCurvature(float alpha, float* atd_results, float* ricci_results) {
    float *d_atd_results, *d_ricci_results;
    cudaMalloc(&d_atd_results, n * sizeof(float));
    cudaMalloc(&d_ricci_results, m * sizeof(float));

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    computeATDandRicciKernel<<<numBlocks, blockSize>>>(n, d_degree, d_neighbors_offset, d_neighbors, 
                                                       d_atd_results, d_ricci_results, alpha);

    cudaMemcpy(atd_results, d_atd_results, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(ricci_results, d_ricci_results, m * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_atd_results);
    cudaFree(d_ricci_results);
}