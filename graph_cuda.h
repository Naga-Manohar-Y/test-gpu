#ifndef GRAPH_CUDA_H
#define GRAPH_CUDA_H

#include "graph.h"

__global__ void floyd_warshall_kernel(float* dist, int k, int n);
__global__ void compute_atd_kernel(float* apsp, ui* neighbors, ept* neighbors_offset, 
                                   float* atd_results, ui n, float alpha);

void computeAPSP_cuda(Graph* graph);
void computeATD_cuda(Graph* graph, float alpha);

#endif // GRAPH_CUDA_H