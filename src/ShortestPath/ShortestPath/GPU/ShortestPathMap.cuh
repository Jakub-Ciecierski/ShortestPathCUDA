#ifndef _SHORTEST_CUDA_H_
#define _SHORTEST_CUDA_H_

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../Macros.h"

#define MAX_GRID_DIM_SIZE 65535

#define MAX_BLOCK_THREAD_COUNT 512

#define MAX_BLOCK_DIM_SIZE 32

#define INF 99999

#define CUDA_ERR(err) (printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__),\
                    exit(EXIT_FAILURE))

__device__
void relax(int* distance, int* previous, int* mask, int v, int u, int weight);


__global__
void shortestPathRelaxationKernel(int* map, int* distance, int* mask, int* previous, 
                                    int dest, int vertexCount, int n, int m, int* done);

__global__
void stopConditionKernel(int* mask, int vertexCount, int n, int m, int* done);

__global__
void shortestPathRelaxationKernelSingle(int* map, int* distance, int* mask, int* previous, 
                                        int dest, int vertexCount, int n, int m, int* done);

/**
* Returns 1 (true) if array is empty - all vertices have been marked 0,
* 0 (false) otherwise
*/
__host__
int isEmpty(int* mask, int n);

__host__
void shortestPathLogic(int* d_map, int* d_distance, int* d_mask, 
                       int* d_previous, int dest, int vertexCount, int n, int m, float* kernelTime);

__host__
void shortestPathInit(int* map, int vertexCount, int n, int m, int source, int dest, 
                        int* distance, int* previous, float* kernelTime);

#endif