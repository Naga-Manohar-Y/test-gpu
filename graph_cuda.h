#ifndef GRAPH_CUDA_H
#define GRAPH_CUDA_H

#include "graph.h"

// it won't show errors on nvcc compiler

__global__ void floyd_warshall_kernel(float* dist, int k, int n);

// __global__ void compute_atd_kernel(float* apsp, ui* neighbors, ept* neighbors_offset, 
//                                    float* atd_results, ui n, float alpha);

__global__ void compute_atd_kernel(float* atd_results, ui n, float alpha);
#endif // GRAPH_CUDA_H