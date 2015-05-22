#include "ShortestPathMap.cuh"

/*
    The usage of atomicMin:
    When atleast two verticies are accessing the same neighbor
    we take the vertex with minimal distance + weight as its previous.
*/
__device__
void relax(int* distance, int* previous, int* mask, int v, int u, int weight){
    if (weight > 0){
        // limit the amount of global memory access
        int dist_v = distance[v];

        if (distance[u] > dist_v + weight){
            atomicMin(&distance[u], dist_v + weight);
            if (dist_v + weight == distance[u]) {   // This might find a weird path, still shortest though
                previous[u] = v;
                mask[u] = 1; 
            }
        }
    }
}

__global__
void shortestPathRelaxationKernel(int* map, int* distance, int* mask,
                                    int* previous, int dest, int vertexCount, int n, int m, int* done)
{
    int col = (blockDim.x * blockIdx.x) + threadIdx.x;
    int row = (blockDim.y * blockIdx.y) + threadIdx.y;

    if (row < n && col < m){
        // main
        int v = row*m + col;
        
        // neighbor
        int u;
        if (mask[v] == 1){
            mask[v] = 0;

            int upper_i = row - 1;
            int upper_j = col;

            int right_i = row;
            int right_j = col + 1;

            int bottom_i = row + 1;
            int bottom_j = col;

            int left_i = row;
            int left_j = col - 1;

            if (upper_i >= 0) {
                u = upper_i*m + upper_j;
                int weight = map[u];
                relax(distance, previous, mask, v, u, weight);
            }

            if (right_j <= m - 1){
                u = right_i*m + right_j;
                int weight = map[u];
                relax(distance, previous, mask, v, u, weight);
            }

            if (bottom_i <= n - 1){
                u = bottom_i*m + bottom_j;
                int weight = map[u];
                relax(distance, previous, mask, v, u, weight);
            }

            if (left_j >= 0){
                u = left_i*m + left_j;
                int weight = map[u];
                relax(distance, previous, mask, v, u, weight);
            }
        }
    }
}

__global__
void stopConditionKernel(int* mask, int vertexCount, int n, int m, int* done)
{
    int col = (blockDim.x * blockIdx.x) + threadIdx.x;
    int row = (blockDim.y * blockIdx.y) + threadIdx.y;

    if (row < n && col < m){
        int v = row*n + col;
        atomicExch(done, mask[v]);
   } 
}


__global__
void shortestPathRelaxationKernelSingle(int* map, int* distance, int* mask,
                                        int* previous, int dest, int vertexCount, int n, int m, int* done)
{
    int col = (blockDim.x * blockIdx.x) + threadIdx.x;
    int row = (blockDim.y * blockIdx.y) + threadIdx.y;

    map[row*n + col] = row*n + col;

    /*
    if (v < vertexCount) {
        while (*done != 0){
            if (mask[v] != 0){

                mask[v] = 0;
                int u = 0;
                for (u = 0; u < vertexCount; u++){
                    int weight = map[vertexCount*v + u];
                    if (weight > 0){
                        if (distance[u] > distance[v] + weight){
                            atomicMin(&distance[u], distance[v] + weight);
                            if (distance[v] + weight == distance[u])
                                previous[u] = v;
                            mask[u] = 1;
                        }
                    }
                }
            }
            atomicMax(done, mask[v]);
        }
    }*/
}



__global__
void test(int* mask, int vertexCount, int n, int m, int* done)
{
    atomicAdd(done, 1);
    int col = (blockDim.x * blockIdx.x) + threadIdx.x;
    int row = (blockDim.y * blockIdx.y) + threadIdx.y;

    if (row < n && col < m){
        int v = row*n + col;
        atomicAdd(done, 1);
    }
}

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
void shortestPathLogic(int* d_map, int* d_distance, int* d_mask, int* d_previous, 
                        int dest, int vertexCount, int n, int m, float* kernelTime)
{
    int i = 0;
    int size = vertexCount*sizeof(int);
    cudaError_t err;

    int* d_done;
    int* ptr_done;

    // flag vector
    int* mask = (int*)malloc(size);

    if ((err = cudaMemcpy(mask, d_mask, size, cudaMemcpyDeviceToHost)) != cudaSuccess) CUDA_ERR(err);
    if ((err = cudaHostAlloc((void**)&d_done, sizeof(int), cudaHostAllocMapped)) != cudaSuccess) CUDA_ERR(err);
    *d_done = 1;

    int gridX = (m + MAX_BLOCK_DIM_SIZE - 1) / MAX_BLOCK_DIM_SIZE;
    int gridY = (n + MAX_BLOCK_DIM_SIZE - 1) / MAX_BLOCK_DIM_SIZE;
    int blockX = MAX_BLOCK_DIM_SIZE;
    int blockY = MAX_BLOCK_DIM_SIZE;

    dim3 DimGrid(gridX, gridY, 1);
    dim3 DimBlock(blockX, blockY, 1);

    printf("DimGrid: x: %d, y: %d, z: %d \n", DimGrid.x, DimGrid.y, DimGrid.z);
    printf("DimBlock: x: %d, y: %d, z: %d \n\n", DimBlock.x, DimBlock.y, DimBlock.z);

    float time;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // SINGLE KERNEL
    /*
    shortestPathRelaxationKernelSingle << <DimGrid, DimBlock >> >(d_map, d_distance, d_mask,
                                                                    d_previous, dest, vertexCount, n, m, d_done);
    int* map = (int*)malloc(size);
    if ((err = cudaMemcpy(map, d_map, size, cudaMemcpyDeviceToHost)) != cudaSuccess) ERR(err);
    int j;
    for (i = 0; i < n; i++){
        for (j = 0; j < n; j++){
            printf("map[%d] = %d\n", i*n + j, map[i*n + j]);
        }
    }
    if ((err = cudaMemcpy(done, d_done, sizeof(int), cudaMemcpyDeviceToHost)) != cudaSuccess) CUDA_ERR(err);
    printf("Count: %d", *done);
    */
    
    
    // STANDARD LOOP KERNEL
    
    while (isEmpty(mask, vertexCount) == 0){

        shortestPathRelaxationKernel << <DimGrid, DimBlock >> >(d_map, d_distance, d_mask,
                                                                d_previous, dest, vertexCount, n, m, d_done);
        cudaDeviceSynchronize();
        if ((err = cudaMemcpy(mask, d_mask, size, cudaMemcpyDeviceToHost)) != cudaSuccess) CUDA_ERR(err);

    }
    
    // SINGLE FLAG PINNED POINTER KERNEL
    
    /*
    if ((err = cudaHostGetDevicePointer(&ptr_done, d_done, 0)) != cudaSuccess) CUDA_ERR(err);
    while (*d_done != 0){
        *d_done = 0;
        shortestPathRelaxationKernel << <DimGrid, DimBlock >> >(d_map, d_distance, d_mask,
                                                                d_previous, dest, vertexCount, n, m, ptr_done);
        //stopConditionKernel << <DimGrid, DimBlock >> >(d_mask, vertexCount, n, m, ptr_done);
        //if ((err = cudaMemcpy(done, d_done, sizeof(int), cudaMemcpyDeviceToHost)) != cudaSuccess) CUDA_ERR(err);
        cudaDeviceSynchronize();
        //printf("Done: %d\n", *d_done);
    }
    */

    // TEST
    /*
    printf("Before Done: %d\n", *d_done);
    cudaHostGetDevicePointer(&ptr_done, d_done, 0);
    test << <DimGrid, DimBlock >> >(d_mask, vertexCount, n, m, d_done);
    cudaDeviceSynchronize();
    printf("After Done: %d\n", *d_done);
    */

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(kernelTime, start, stop);

    printf("Device finished after: %f ms\n", *kernelTime);
}

__host__
void shortestPathInit(int* map, int vertexCount, int n, int m, int source, int dest, int* distance, int* previous, float* kernelTime)
{
    /********** INITIATE CPU ARRAYS **********/
    int size = vertexCount*sizeof(int);

    // flag vector
    int* mask = (int*)malloc(size);
    if (!mask) ERR("mask malloc()");

    int i = 0;
    for (i = 0; i < vertexCount; i++)
    {
        distance[i] = INF;
        mask[i] = 0;
        previous[i] = -1;
    }

    mask[source] = 1;
    distance[source] = 0;

    /*********** INITIATE DEVICE ARRAYS ***********/
    int* d_map;
    int* d_distance;
    int* d_mask;
    int* d_previous;

    cudaError_t err;

    /*********** MEMORY ALLOCATION ***********/
    if ((err = cudaMalloc((void**)&d_map, size)) != cudaSuccess) CUDA_ERR(err);
    if ((err = cudaMalloc((void**)&d_distance, size)) != cudaSuccess) CUDA_ERR(err);
    if ((err = cudaMalloc((void**)&d_mask, size)) != cudaSuccess) CUDA_ERR(err);
    if ((err = cudaMalloc((void**)&d_previous, size)) != cudaSuccess) CUDA_ERR(err);

    /*********** COPY MEMORY TO DEVICE ***********/
    if ((err = cudaMemcpy(d_map, map, size, cudaMemcpyHostToDevice)) != cudaSuccess) CUDA_ERR(err);
    if ((err = cudaMemcpy(d_distance, distance, size, cudaMemcpyHostToDevice)) != cudaSuccess) CUDA_ERR(err);
    if ((err = cudaMemcpy(d_mask, mask, size, cudaMemcpyHostToDevice)) != cudaSuccess) CUDA_ERR(err);
    if ((err = cudaMemcpy(d_previous, previous, size, cudaMemcpyHostToDevice)) != cudaSuccess) CUDA_ERR(err);

    // work ...
    shortestPathLogic(d_map, d_distance, d_mask, d_previous, dest, vertexCount, n, m, kernelTime);

    /*********** COPY MEMORY BACK TO HOST ***********/
    if ((err = cudaMemcpy(distance, d_distance, size, cudaMemcpyDeviceToHost)) != cudaSuccess) CUDA_ERR(err);
    if ((err = cudaMemcpy(previous, d_previous, size, cudaMemcpyDeviceToHost)) != cudaSuccess) CUDA_ERR(err);

    /*********** FREE MEMORY ***********/
    if ((err = cudaFree(d_map)) != cudaSuccess) CUDA_ERR(err);
    if ((err = cudaFree(d_distance)) != cudaSuccess) CUDA_ERR(err);
    if ((err = cudaFree(d_mask)) != cudaSuccess) CUDA_ERR(err);
    if ((err = cudaFree(d_previous)) != cudaSuccess) CUDA_ERR(err);
}