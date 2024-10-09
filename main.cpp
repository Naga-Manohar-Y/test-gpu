#include "graph.h"
#include <iostream>
#include <algorithm>

int main() {
    const char* directory = "."; // This means the current directory
    
    try {
        Graph g(directory);
        
        // Read the text file
        g.readTextFile("test_gh2.txt");
        
        // Write to a binary file
        g.writeBinaryFile("test2.bin");
        
        // Read the binary file back (to test)
        Graph g2(directory);
        g2.readBinaryFile("test2.bin");
        
        // APSP is computed automatically in readBinaryFile

        // Print a sample of APSP results
        std::cout << "APSP Results (full matrix):" << std::endl;
        for (unsigned int i = 0; i < g2.getN(); i++) {
            for (unsigned int j = 0; j < g2.getN(); j++) {
                std::cout << g2.getAPSP(i, j) << "\t";
            }
            std::cout << std::endl;
        }
        
        // Print a sample of APSP results
        // std::cout << "APSP Results (first 5x5 submatrix):" << std::endl;
        // for (unsigned int i = 0; i < std::min(5u, g2.getN()); i++) {
        //     for (unsigned int j = 0; j < std::min(5u, g2.getN()); j++) {
        //         std::cout << g2.getAPSP(i, j) << "\t";
        //     }
        //     std::cout << std::endl;
        // }

        // Clean up is handled automatically by destructors


        // Compute ATD
        float alpha = 0.5; // Set your desired alpha value
        g2.computeATD(alpha);

        // Check if ATD computation was successful
        if (g2.getATD(0, 0) == 0.0f) {
            std::cout << "ATD computation successful." << std::endl;
        } else {
            std::cerr << "ATD computation may have failed." << std::endl;
        }



        // Print sample ATD results
        std::cout << "ATD Results (first 5x5 submatrix):" << std::endl;
        for (unsigned int i = 0; i < std::min(5u, g2.getN()); i++) {
            for (unsigned int j = 0; j < std::min(5u, g2.getN()); j++) {
                std::cout << g2.getATD(i, j) << "\t";
            }
            std::cout << std::endl;
        }


    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}