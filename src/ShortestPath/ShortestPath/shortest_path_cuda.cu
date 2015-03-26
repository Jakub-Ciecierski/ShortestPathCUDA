#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "graph.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define INF 99999

__host__
void matrixMultiplication(float* A, float* B, float* C, int n)
{
    // size in bytes
    int size = n*n*sizeof(float);
    // device vectors
    float *d_A, *d_B, *d_C;

    cudaError_t err;

    // (address of the pointer, size in bytes)
    err = cudaMalloc((void**)&d_A, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n",
            cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void**)&d_B, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n",
            cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void**)&d_C, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n",
            cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    printf("Alloced memory on device \n");

    // (dest, source, size in bytes, type of transfer)
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    printf("Copied memory to device \n");
    // The matrix C is devided into (n / tile_width) smaller submatrices blocks
    // of dim (tile_width x tile_width) each
    // x, y, z
    dim3 DimGrid(n / TILE_WIDTH, n / TILE_WIDTH, 1);
    dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    printf("DimGrid: x: %d, y: %d, z: %d \n", DimGrid.x, DimGrid.y, DimGrid.z);
    printf("DimBlock: x: %d, y: %d, z: %d \n", DimBlock.x, DimBlock.y, DimBlock.z);

    //call the kernel function
    matrixMultiplicationKernel << <DimGrid, DimBlock >> >(d_A, d_B, d_C, n);

    // get the computed vector back to host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

__host__
void shortestPath(graph_t* graph, int source, int* distance,  int* previous)
{
    int n = graph->num_vertices;

    int* local_distance = (int*)malloc(n * sizeof(int));

    int i = 0;
    for (i = 0; i < n; i++)
    {
        distance[i] = INF;
        local_distance[i] = INF;
        previous[i] = -1;
    }
    distance[source] = INF;
    local_distance[source] = INF;

    graph_t* d_graph;
    int* d_distance;
    int* d_local_distance;
    int* d_previous;

    cudaError_t err;

    // (address of the pointer, size in bytes)
    err = cudaMalloc((void**)&d_A, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n",
            cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
}

__host__
int main()
{
    graph_t* graph = createGraph(5);
    addEdge(graph, 0, 3, 2);
    addEdge(graph, 0, 4, 1);
    addEdge(graph, 1, 2, 1);
    addEdge(graph, 2, 3, 3);
    addEdge(graph, 4, 2, 3);
    addEdge(graph, 3, 1, 2);

    showGraph(graph);

    int num_vertices = graph->num_vertices;
    int* distance = (int*)malloc(num_vertices * sizeof(int));
    int* previous = (int*)malloc(num_vertices * sizeof(int));

    clock_t start, end;
    double delta;
    start = clock();


    end = clock();
    printf("Start: %d End: %d \n", start, end);
    delta = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Finished after %f \n", delta);

    return EXIT_SUCCESS;
}