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

    float* apsp;

    float* weights;  // Array to store edge weights
    
    


    void readDIMACS2Text(const char* filepath);
    void readRawSNAPText(const char* filepath);

    // GPU memory pointers
    ept *d_neighbors_offset;
    ui *d_neighbors;
    ui *d_degree;

    float* d_weights;  // GPU memory for weights

    float *d_apsp;

    float* atd_results; // Average transport distance
    float* d_atd_results = nullptr ;// my

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

    // APSP calculation method
    void assignEdgeWeights();
    void computeAPSP();

    float getAPSP(unsigned int i, unsigned int j) const {
        return apsp[i * n + j];
    }

    void computeATD(float alpha);
    void computeATDCPU(float alpha);
    float getATD(unsigned int i, unsigned int j) const;


    // Getter methods for n and m
    ui getN() const { return n; }
    ept getM() const { return m; }
};

#endif