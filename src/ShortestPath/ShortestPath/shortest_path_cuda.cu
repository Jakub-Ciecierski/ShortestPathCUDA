#include "shortest_path_cuda.cuh"

__device__
int getWeight(int* graphMatrix, int n, int v, int u)
{
    return graphMatrix[n*v + u];
}

__global__
void shortestPathRelaxationKernel(int* graphMatrix, int* distance,
                                    int* local_distance, int* mask, 
                                    int* previous, int* local_previous,
                                    int dest, int n)
{
    dest++;
    int v = threadIdx.x + blockDim.x*blockIdx.x;

    // if tid is in bound of the vector and
    // if tid is in the mask
    if (v < n && mask[v] == 1){

        // remove this vertex from mask
        mask[v] = 0;
        int u = 0;
        for (u = 0; u < n; u++){
            // get weight of each other vertex
            int weight = graphMatrix[n*v + u];
            // if it is a neighbor
            if (weight > 0){
                // Local distance relaxation
                if (distance[u] > distance[v] + weight){
                    
                    
                    atomicMin(&distance[u], distance[v] + weight);
                    if (distance[v] + weight == distance[u])
                        previous[u] = v;
                    mask[u] = 1; 
                }
            }
        }
    }
}

__global__
void shortestPathUpdateDistanceKernel(int* graphMatrix, int* distance,
                                        int* local_distance, int* mask, 
                                        int* previous, int* local_previous,
                                        int dest, int n)
{
    int v = threadIdx.x + blockDim.x*blockIdx.x;
    if (v < n){
        if (distance[v] > local_distance[v]){
            distance[v] = local_distance[v];
            previous[v] = local_previous[v];
            mask[v] = 1;
        }
        local_distance[v] = distance[v];
    }
}

/**
 * Returns 1 (true) if array is empty - all vertices have been marked 0,
 * 0 (false) otherwise
 */
__host__ 
int isEmpty(int* mask, int n)
{
    int i = 0;
    for (i = 0; i < n; i++){
        if (mask[i] == 1)
            return 0;
    }
    return 1;
}

__host__ 
void shortestPathLogic(int* d_graphMatrix, int* d_distance,
                       int* d_local_distance, int* d_mask, 
                       int* d_previous, int* d_local_previous,
                       int dest, int n)
{
    int size = n*sizeof(int);
    int sizeGraph= n*sizeof(int);

    int* mask = (int*)malloc(size);
    int* local_distance = (int*)malloc(size);
    int* graphMatrix = (int*)malloc(sizeGraph);

    int* previous = (int*)malloc(size); //todo

    cudaError_t err;
    if ((err = cudaMemcpy(mask, d_mask, size, cudaMemcpyDeviceToHost)) != cudaSuccess) ERR(err);
    if ((err = cudaMemcpy(graphMatrix, d_graphMatrix, sizeGraph, cudaMemcpyDeviceToHost)) != cudaSuccess) ERR(err);
    if ((err = cudaMemcpy(previous, d_previous, size, cudaMemcpyDeviceToHost)) != cudaSuccess) ERR(err); // todo
    
    int i = 0;

    // TODO check if gridX is greater than MAX_GRID_DIM_SIZE
    int gridX = ((n - 1) / MAX_BLOCK_DIM_SIZE) + 1;
    int blockX = MAX_BLOCK_DIM_SIZE;

    dim3 DimGrid(gridX, 1, 1);
    dim3 DimBlock(blockX, 1, 1);

    printf("DimGrid: x: %d, y: %d, z: %d \n", DimGrid.x, DimGrid.y, DimGrid.z);
    printf("DimBlock: x: %d, y: %d, z: %d \n", DimBlock.x, DimBlock.y, DimBlock.z);

    while (isEmpty(mask, n) == 0){

        shortestPathRelaxationKernel << <DimGrid, DimBlock >> >(d_graphMatrix, d_distance, 
                                                                d_local_distance, d_mask, 
                                                                d_previous, d_local_previous, 
                                                                dest, n);
        /*
        shortestPathUpdateDistanceKernel << <DimGrid, DimBlock >> >(d_graphMatrix, d_distance,
                                                                d_local_distance, d_mask, 
                                                                d_previous, d_local_previous,
                                                                dest, n);
        */
        if ((err = cudaMemcpy(mask, d_mask, size, cudaMemcpyDeviceToHost)) != cudaSuccess) ERR(err);

    }
}

__host__
void shortestPathInit(int* graphMatrix, int n, int source, int dest, int* distance, int* previous)
{
    /********** INITIATE CPU ARRAYS **********/
    int graphSize = n*n*sizeof(int);
    int size = n*sizeof(int);
    // distance vector
    int* local_distance = (int*)malloc(size);
    int* local_previous = (int*)malloc(size);
    // flag vector
    int* mask = (int*)malloc(size);

    int i = 0;
    for (i = 0; i < n; i++)
    {
        distance[i] = INF;
        local_distance[i] = INF;
        mask[i] = 0;
        previous[i] = -1;
        local_previous[i] - 1;
    }

    // mark source as visited
    mask[source] = 1;
    distance[source] = 0;
    local_distance[source] = 0;

    /*********** INITIATE DEVICE ARRAYS ***********/
    int* d_graphMatrix;
    int* d_distance;
    int* d_local_distance;
    int* d_mask;
    int* d_previous;
    int* d_local_previous;

    cudaError_t err;

    /*********** MEMORY ALLOCATION ***********/
    if ((err = cudaMalloc((void**)&d_graphMatrix, graphSize)) != cudaSuccess) ERR(err);
    if ((err = cudaMalloc((void**)&d_distance, size)) != cudaSuccess) ERR(err);
    if ((err = cudaMalloc((void**)&d_local_distance, size)) != cudaSuccess) ERR(err);
    if ((err = cudaMalloc((void**)&d_mask, size)) != cudaSuccess) ERR(err);
    if ((err = cudaMalloc((void**)&d_previous, size)) != cudaSuccess) ERR(err);
    if ((err = cudaMalloc((void**)&d_local_previous, size)) != cudaSuccess) ERR(err);

    /*********** COPY MEMORY ***********/
    if ((err = cudaMemcpy(d_graphMatrix, graphMatrix, graphSize, cudaMemcpyHostToDevice)) != cudaSuccess) ERR(err);
    if ((err = cudaMemcpy(d_distance, distance, size, cudaMemcpyHostToDevice)) != cudaSuccess) ERR(err);
    if ((err = cudaMemcpy(d_local_distance, local_distance, size, cudaMemcpyHostToDevice)) != cudaSuccess) ERR(err);
    if ((err = cudaMemcpy(d_mask, mask, size, cudaMemcpyHostToDevice)) != cudaSuccess) ERR(err);
    if ((err = cudaMemcpy(d_previous, previous, size, cudaMemcpyHostToDevice)) != cudaSuccess) ERR(err);
    if ((err = cudaMemcpy(d_local_previous, local_previous, size, cudaMemcpyHostToDevice)) != cudaSuccess) ERR(err);

    // work ...
    shortestPathLogic(d_graphMatrix, d_distance, d_local_distance, d_mask, d_previous, d_local_previous, dest, n);

    if ((err = cudaMemcpy(distance, d_distance, size, cudaMemcpyDeviceToHost)) != cudaSuccess) ERR(err);
    if ((err = cudaMemcpy(previous, d_previous, size, cudaMemcpyDeviceToHost)) != cudaSuccess) ERR(err);

    /*********** FREE MEMORY ***********/
    if ((err = cudaFree(d_graphMatrix)) != cudaSuccess) ERR(err);
    if ((err = cudaFree(d_distance)) != cudaSuccess) ERR(err);
    if ((err = cudaFree(d_local_distance)) != cudaSuccess) ERR(err);
    if ((err = cudaFree(d_mask)) != cudaSuccess) ERR(err);
    if ((err = cudaFree(d_previous)) != cudaSuccess) ERR(err);
    if ((err = cudaFree(d_local_previous)) != cudaSuccess) ERR(err);
}