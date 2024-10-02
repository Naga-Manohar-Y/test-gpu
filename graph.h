#ifndef _GRAPH_H_
#define _GRAPH_H_

#include <string>
#include <vector>

typedef unsigned int ui;
typedef long long ept;

class Graph {
private:
    std::string dir; // input graph directory
    ui n; // number of nodes of the graph
    ept m; // number of edges of the graph

    ept *neighbors_offset; // offset of neighbors of nodes
    ui *neighbors; // adjacent ids of edges
    ui *degree; // degree of each node
    ui *reverse;

    void readDIMACS2Text(const char* filepath);
    void readRawSNAPText(const char* filepath);

    // GPU memory pointers
    ept *d_neighbors_offset;
    ui *d_neighbors;
    ui *d_degree;

public:
    Graph(const char *_dir);
    ~Graph();

    void readTextFile(const char* filepath);
    void writeBinaryFile(const char* filepath);
    void readBinaryFile(const char* filepath);

    // GPU memory management methods
    void mallocGraphGPUMemory();
    void freeGraphGPUMemory();
    void copyToGPU();
    void copyFromGPU();

    // Ricci curvature calculation method
    void computeRicciCurvature(float alpha, float* atd_results, float* ricci_results);

    // Getter methods for n and m
    ui getN() const { return n; }
    ept getM() const { return m; }
};

#endif