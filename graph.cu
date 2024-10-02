#define _CRT_SECURE_NO_WARNINGS
#include "graph.h"
#include <cuda_runtime.h>

// #include <cuda_runtime.h>
#include <cassert>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <sstream>
#include <map>
#include <cstring>
#include <iostream>

using ui = unsigned int;
const ui FILELEN = 1024;

static int fileSuffixPos(char* filepath) {
    // finds the file extention in the file name
	int j = strlen(filepath) - 1;
	while (filepath[j] != '.')
		j--;
	return j + 1;
}

static FILE *open_file(const char *file_name, const char *mode) {
    // wrapper function for 'fopen' 
	FILE *f = fopen(file_name, mode);
	if (f == nullptr) {
		printf("Can not open file: %s\n", file_name);
		exit(1);
	}

	return f;
}

static std::string integer_to_string(long long number) {
    // converts long integer to formatted string  with ',' separators
	std::vector<ui> sequence;
	if (number == 0) sequence.push_back(0);
	while (number > 0) {
		sequence.push_back(number % 1000);
		number /= 1000;
	}

	char buf[5];
	std::string res;
	for (unsigned int i = sequence.size(); i > 0; i--) {
		if (i == sequence.size()) sprintf(buf, "%u", sequence[i - 1]);
		else sprintf(buf, ",%03u", sequence[i - 1]);
		res += std::string(buf);
	}
	return res;
}
//----------------------------------------------------------------------

Graph::Graph(const char *_dir) : dir(_dir), n(0), m(0), neighbors_offset(nullptr), neighbors(nullptr), degree(nullptr),reverse(nullptr){
}

Graph::~Graph() {
    delete[] neighbors_offset;
    delete[] neighbors;
    delete[] degree;
    delete[] reverse;
}

void Graph::readTextFile(const char* filepath) {
    // Detect file format and call appropriate function
    if (strstr(filepath, ".dim")) {
        readDIMACS2Text(filepath);
    } else {
        readRawSNAPText(filepath);
    }
}

void Graph::writeBinaryFile(const char* filepath) {
    // ... (implementation remains the same)
    FILE *f = open_file(filepath, "wb");
	ui tt = sizeof(ui);
	fwrite(&tt, sizeof(ui), 1, f); //length of ui
	fwrite(&n, sizeof(ui), 1, f);
	fwrite(&m, sizeof(ui), 1, f);
    std::cout << "Writing n: " << n << ", m: " << m << std::endl;

	ui *degree = new ui[n];
	for (ui i = 0; i < n; i++)
		degree[i] = neighbors_offset[i + 1] - neighbors_offset[i];
	fwrite(degree, sizeof(ui), n, f);
	fwrite(neighbors, sizeof(ui), m, f);
	fclose(f);
    delete[] degree;
}


void Graph::readBinaryFile(const char* filepath) {
    FILE* f = fopen(filepath, "rb");
    if (f == NULL) {
        std::cerr << "Error: Could not open file for reading." << std::endl;
        return;
    }

    // Read size of ui
    ui tt;
    fread(&tt, sizeof(ui), 1, f);
    // Ensure that sizeof(ui) matches the value read
    if (tt != sizeof(ui)) {
        std::cerr << "Error: ui size mismatch." << std::endl;
        fclose(f);
        return;
    }

    // Read number of nodes and edges
    fread(&n, sizeof(ui), 1, f);
    fread(&m, sizeof(ui), 1, f);
    std::cout << "Read n: " << n << ", m: " << m << std::endl;

    // Allocate memory for degree array
    ui* degree = new ui[n];
    fread(degree, sizeof(ui), n, f);

    // Print the degree array
    std::cout << "Degree array:" << std::endl;
    for (ui i = 0; i < n; i++) {
        std::cout << degree[i] << " ";
    }
    std::cout << std::endl;

    // Allocate memory for neighbors_offset and neighbors arrays
    neighbors_offset = new ept[n + 1];
    neighbors = new ui[m];

    // Compute neighbors_offset from degree array
    ui j = 0;
    for (ui i = 0; i < n; i++) {
        neighbors_offset[i] = j;
        j += degree[i];
    }
    neighbors_offset[n] = j;

    // Print the neighbors_offset array
    std::cout << "neighbors_offset array:" << std::endl;
    for (ui i = 0; i <= n; i++) {
        std::cout << neighbors_offset[i] << " ";
    }
    std::cout << std::endl;

    // Read neighbors array
    fread(neighbors, sizeof(ui), m, f);

    // Print the neighbors array
    std::cout << "neighbors array:" << std::endl;
    for (ui i = 0; i < m; i++) {
        std::cout << neighbors[i] << " ";
    }
    std::cout << std::endl;

	// After reading, allocate GPU memory and copy data

	mallocGraphGPUMemory();
    copyToGPU();

    delete[] degree;
    fclose(f);
}



void Graph::readDIMACS2Text(const char* filepath) {
    // Implementation of readDIMACS2Text
    std::ifstream infile;
	const int SZBUF = 99999999;
	char *buf = new char[SZBUF];
	std::vector<std::pair<ui, ui> > epairs;
	std::vector<ui> nodes;
	//FILE *f = Utility::open_file(filepath, "r");
	infile.open(filepath, std::ios::in);
	if (!infile.is_open()) {
		fprintf(stderr, "can not find file %s\n", filepath);
		exit(1);
	}

	infile.getline(buf, SZBUF);
	while (buf[0] == '%') infile.getline(buf, SZBUF);

	std::stringstream ss(buf);
	int fmt = 0;
	ss >> n >> m >> fmt;
	if (fmt != 0){
		printf("Format of %s is not supported yet\n", filepath);
		exit(0);
	}
	m *= 2;
	neighbors_offset = new ept[n + 1];
	neighbors = new ui[m];
	reverse = new ui[m];
	ui j = 0;
	for (ui u = 0; u < n; u++) {
		neighbors_offset[u] = j;
		infile.getline(buf, SZBUF);
		std::stringstream ss(buf);
		int nei;
		while (ss >> nei) {
			//printf("%d ", nei);
			if ((nei - 1) != u) {
				neighbors[j] = nei - 1;
				reverse[j] = u;
				j++;
				//if (j==745)
				//	printf("pause\n");
			}
		}
		//printf("\n");
		std::sort(neighbors + neighbors_offset[u], neighbors + j);
	}
	neighbors_offset[n] = j;	
	assert(j == m);
    printf("read from text file\n");
	printf("n:%u m:%u\n",n,m/2);
}

void Graph::readRawSNAPText(const char* filepath) {
    // Implementation of readRawSNAPText
    std::ifstream infile;
	char buf[1024];
	std::vector<std::pair<ui, ui> > epairs;
	std::vector<ui> nodes;
	//FILE *f = Utility::open_file(filepath, "r");
	infile.open(filepath, std::ios::in);
	if (!infile.is_open()) {
		fprintf(stderr, "can not find file %s\n", filepath);
		exit(1);
	}
	int max_id = 0;
	int from, to;
	while (infile.getline(buf, 1024)) {
		char *p = buf;
		while (*p == ' ' && *p != '\0') p++;
		if (*p == '#' || *p == '\0') continue;
		std::stringstream ss(buf);
		ss >> from >> to;
		if (from != to) {
			epairs.push_back(std::make_pair(from, to));
			epairs.push_back(std::make_pair(to, from));
			nodes.push_back(from);
			nodes.push_back(to);
		}
	}
	infile.close();

	sort(nodes.begin(), nodes.end());
	nodes.erase(unique(nodes.begin(), nodes.end()), nodes.end());

	sort(epairs.begin(), epairs.end());
	epairs.erase(unique(epairs.begin(), epairs.end()), epairs.end());

	ui contn = 1;
	std::map<ui, ui> idmp;
	for (ui i = 0; i < nodes.size(); i++) {
		idmp[nodes[i]] = i;
		if (nodes[i] != i) {
			contn = 0;
		}
	}
	if (contn == 0) printf("Node ids are not preserved! \n");

	n = nodes.size();
	m = epairs.size();
	printf("n = %s, (undirected) m = %s\n",
		integer_to_string(n).c_str(),
		integer_to_string(m / 2).c_str());

	neighbors_offset = new ept[n + 1];
	neighbors = new ui[m];
	reverse = new ui[m];
	ui j = 0;
	for (ui i = 0; i < n; i++) {
		neighbors_offset[i] = j;
		while (j < m && epairs[j].first == nodes[i]) {
			neighbors[j] = idmp[epairs[j].second];
			reverse[j] = i;
			++j;
		}
	}
	neighbors_offset[n] = j;
}

void Graph::mallocGraphGPUMemory() {
    cudaMalloc(&d_neighbors_offset, (n + 1) * sizeof(ept));
    cudaMalloc(&d_neighbors, m * sizeof(ui));
    cudaMalloc(&d_degree, n * sizeof(ui));
}

void Graph::freeGraphGPUMemory() {
    cudaFree(d_neighbors_offset);
    cudaFree(d_neighbors);
    cudaFree(d_degree);
}

void Graph::copyToGPU() {
    cudaMemcpy(d_neighbors_offset, neighbors_offset, (n + 1) * sizeof(ept), cudaMemcpyHostToDevice);
    cudaMemcpy(d_neighbors, neighbors, m * sizeof(ui), cudaMemcpyHostToDevice);
    cudaMemcpy(d_degree, degree, n * sizeof(ui), cudaMemcpyHostToDevice);
}

void Graph::copyFromGPU() {
    cudaMemcpy(neighbors_offset, d_neighbors_offset, (n + 1) * sizeof(ept), cudaMemcpyDeviceToHost);
    cudaMemcpy(neighbors, d_neighbors, m * sizeof(ui), cudaMemcpyDeviceToHost);
    cudaMemcpy(degree, d_degree, n * sizeof(ui), cudaMemcpyDeviceToHost);
}