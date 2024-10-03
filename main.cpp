#include "graph.h"
#include <iostream>
#include <algorithm>

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
    
    // APSP is computed automatically in readBinaryFile
    
    // Print a sample of APSP results
    std::cout << "APSP Results (first 5x5 submatrix):" << std::endl;
    for (unsigned int i = 0; i < std::min(5u, g2.getN()); i++) {
        for (unsigned int j = 0; j < std::min(5u, g2.getN()); j++) {
            std::cout << g2.getAPSP(i, j) << "\t";
        }
        std::cout << std::endl;
    }

    // Clean up
    g2.freeGraphGPUMemory();
    
    return 0;
}