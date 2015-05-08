#ifndef _SHORTEST_CUDA_H_
#define _SHORTEST_CUDA_H_

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define MAX_GRID_DIM_SIZE 65535

#define MAX_BLOCK_THREAD_COUNT 512

#define MAX_BLOCK_DIM_SIZE 512

#define INF 99999

#define ERR(err) (printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__),\
                    exit(EXIT_FAILURE))

__device__
int getWeight(int* graphMatrix, int n, int v, int u);

__global__
void shortestPathRelaxationKernel(int* graphMatrix, int* distance,
                                    int* local_distance, int* mask, int* previous, int dest, int n);

__global__
void shortestPathUpdateDistanceKernel(int* graphMatrix, int* distance,
                                        int* local_distance, int* mask, int* previous, int dest,int n);

/**
* Returns 1 (true) if array is empty - all vertices have been marked 0,
* 0 (false) otherwise
*/
__host__
int isEmpty(int* mask, int n);


__host__
void shortestPathLogic(int* d_graphMatrix, int* d_distance,
                        int* d_local_distance, int* d_mask, int* d_previous, int dest,int n);

__host__
void shortestPathInit(int* graphMatrix, int n, int source, int dest, int* distance, int* previous);

#endif