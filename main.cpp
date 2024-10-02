#include "graph.h"
#include <iostream>

int main() {
    const char* directory = "."; // This means the current directory
    Graph g(directory);
    
    // Read the text file
    g.readTextFile("test_gh.txt");
    
    // Write to a binary file
    g.writeBinaryFile("test.bin");
    
    // Read the binary file back (to test)
    Graph g2(directory);
    g2.readBinaryFile("test.bin");
    
    // Compute Ricci curvature
    float* atd_results = new float[g2.n];
    float* ricci_results = new float[g2.m];
    float alpha = 0.5; // Set your desired alpha value
    g2.computeRicciCurvature(alpha, atd_results, ricci_results);

    // Print results (first few elements)
    std::cout << "ATD Results (first 10):" << std::endl;
    for (int i = 0; i < std::min(10u, g2.n); i++) {
        std::cout << "Node " << i << ": " << atd_results[i] << std::endl;
    }

    std::cout << "\nRicci Curvature Results (first 10):" << std::endl;
    for (int i = 0; i < std::min(10u, g2.m); i++) {
        std::cout << "Edge " << i << ": " << ricci_results[i] << std::endl;
    }

    // Clean up
    delete[] atd_results;
    delete[] ricci_results;
    g2.freeGraphGPUMemory();
    
    return 0;
}