#include "ShortestPath.cuh"

__device__
int getWeight(int* graphMatrix, int n, int v, int u)
{
    return graphMatrix[n*v + u];
}

__global__
void shortestPathRelaxationKernel(int* graphMatrix, int* distance, int* mask, 
                                    int* previous, int dest, int n)
{
    int v = threadIdx.x + blockDim.x*blockIdx.x;

    int left_nb, right_nb, bottom_nb, upper_nb;

    // if tid is in bound of the vector and
    // if tid is in the mask
    if (v < n && mask[v] == 1){
        // remove this vertex from mask
        mask[v] = 0;
        int u = 0;

        /*
        int left_nb = v*n - 1;
        if (left_nb >= 0){
        //if (v%n != 0){
            //int left_nb = v*n - 1;
            u = left_nb;
            int weight = graphMatrix[u];
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

        int right_nb = v*n + 1;
        if (right_nb < n){
        //if ((v + 1) % n != 0){
            //int right_nb = v*n + 1;
            u = right_nb;
            int weight = graphMatrix[u];
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
        
        int bottom_nb = v*n + 3;
        if (bottom_nb < n){//if ((float)(n - v) > sqrtf(n)){
            //int bottom_nb = v*n + 3;
            u = bottom_nb;
            int weight = graphMatrix[u];
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

        int upper_nb = v*n - 3;
        if (upper_nb >= 0){//if ((float)v >= sqrtf(n)){
            //int upper_nb = v*n - 3;
            u = upper_nb;
            int weight = graphMatrix[u];
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
        */
        
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
void shortestPathRelaxationKernelSingle(int* graphMatrix, int* distance, int* mask,
                                        int* previous, int dest, int n, int* done)
{
    int v = threadIdx.x + blockDim.x*blockIdx.x;

    if (v < n) {
        while (*done != 0){
            if (mask[v] != 0){

                mask[v] = 0;
                int u = 0;
                for (u = 0; u < n; u++){
                    int weight = graphMatrix[n*v + u];
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
    }
}


__global__
void shortestPathRelaxationKernelTest(int* graphMatrix, int* distance, int* mask,
                                        int* previous, int dest, int n, int* done)
{
    int v = threadIdx.x + blockDim.x*blockIdx.x;

    if (v < n) {
        while (*done == 1){
            mask[v] = 0;
            
            atomicMax(done, mask[v]);
            
        }
    }
}

/*
__global__
void shortestPathUpdateDistanceKernel(int* graphMatrix, int* distance, int* mask, 
                                        int* previous, int dest, int n)
{
    int v = threadIdx.x + blockDim.x*blockIdx.x;
    if (v < n){
        if (distance[v] > local_distance[v]){
            distance[v] = local_distance[v];
            mask[v] = 1;
        }
        local_distance[v] = distance[v];
    }
}*/


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
void shortestPathLogic(int* d_graphMatrix, int* d_distance, int* d_mask, 
                       int* d_previous, int dest, int n, float* kernelTime)
{
    int i = 0;
    int size = n*sizeof(int);
    cudaError_t err;

    int* done = (int*)malloc(sizeof(int));
    *done = 1;
    int* d_done;

    // flag vector
    int* mask = (int*)malloc(size);

    if ((err = cudaMalloc((void**)&d_done, sizeof(int))) != cudaSuccess) ERR(err);

    if ((err = cudaMemcpy(mask, d_mask, size, cudaMemcpyDeviceToHost)) != cudaSuccess) ERR(err);
    if ((err = cudaMemcpy(d_done, done, sizeof(int), cudaMemcpyHostToDevice)) != cudaSuccess) ERR(err);
    

    // TODO check if gridX is greater than MAX_GRID_DIM_SIZE
    int gridX = ((n - 1) / MAX_BLOCK_DIM_SIZE) + 1;
    int blockX = MAX_BLOCK_DIM_SIZE;

    dim3 DimGrid(gridX, 1, 1);
    dim3 DimBlock(blockX, 1, 1);

    printf("DimGrid: x: %d, y: %d, z: %d \n", DimGrid.x, DimGrid.y, DimGrid.z);
    printf("DimBlock: x: %d, y: %d, z: %d \n\n", DimBlock.x, DimBlock.y, DimBlock.z);

    float time;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    /*
    shortestPathRelaxationKernelTest << <DimGrid, DimBlock >> >(d_graphMatrix, d_distance, d_mask,
                                                                    d_previous, dest, n, d_done);
    if ((err = cudaMemcpy(done, d_done, sizeof(int), cudaMemcpyDeviceToHost)) != cudaSuccess) ERR(err);
    printf("Count: %d", *done);
    */
    
    while (isEmpty(mask, n) == 0){

        shortestPathRelaxationKernel << <DimGrid, DimBlock >> >(d_graphMatrix, d_distance, d_mask, 
                                                                d_previous, dest, n);
        if ((err = cudaMemcpy(mask, d_mask, size, cudaMemcpyDeviceToHost)) != cudaSuccess) ERR(err);

    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(kernelTime, start, stop);
}

__host__
void shortestPathInit(int* graphMatrix, int n, int source, int dest, int* distance, int* previous, float* kernelTime)
{
    /********** INITIATE CPU ARRAYS **********/
    int graphSize = n*n*sizeof(int);
    int size = n*sizeof(int);

    // flag vector
    int* mask = (int*)malloc(size);

    int i = 0;
    for (i = 0; i < n; i++)
    {
        distance[i] = INF;
        mask[i] = 0;
        previous[i] = -1;
    }

    // mark source as visited
    mask[source] = 1;
    distance[source] = 0;

    /*********** INITIATE DEVICE ARRAYS ***********/
    int* d_graphMatrix;
    int* d_distance;
    int* d_mask;
    int* d_previous;

    cudaError_t err;

    /*********** MEMORY ALLOCATION ***********/
    if ((err = cudaMalloc((void**)&d_graphMatrix, graphSize)) != cudaSuccess) ERR(err);
    if ((err = cudaMalloc((void**)&d_distance, size)) != cudaSuccess) ERR(err);
    if ((err = cudaMalloc((void**)&d_mask, size)) != cudaSuccess) ERR(err);
    if ((err = cudaMalloc((void**)&d_previous, size)) != cudaSuccess) ERR(err);

    /*********** COPY MEMORY TO DEVICE ***********/
    if ((err = cudaMemcpy(d_graphMatrix, graphMatrix, graphSize, cudaMemcpyHostToDevice)) != cudaSuccess) ERR(err);
    if ((err = cudaMemcpy(d_distance, distance, size, cudaMemcpyHostToDevice)) != cudaSuccess) ERR(err);
    if ((err = cudaMemcpy(d_mask, mask, size, cudaMemcpyHostToDevice)) != cudaSuccess) ERR(err);
    if ((err = cudaMemcpy(d_previous, previous, size, cudaMemcpyHostToDevice)) != cudaSuccess) ERR(err);

    // work ...
    shortestPathLogic(d_graphMatrix, d_distance, d_mask, d_previous, dest, n, kernelTime);

    /*********** COPY MEMORY BACK TO HOST ***********/
    if ((err = cudaMemcpy(distance, d_distance, size, cudaMemcpyDeviceToHost)) != cudaSuccess) ERR(err);
    if ((err = cudaMemcpy(previous, d_previous, size, cudaMemcpyDeviceToHost)) != cudaSuccess) ERR(err);

    /*********** FREE MEMORY ***********/
    if ((err = cudaFree(d_graphMatrix)) != cudaSuccess) ERR(err);
    if ((err = cudaFree(d_distance)) != cudaSuccess) ERR(err);
    if ((err = cudaFree(d_mask)) != cudaSuccess) ERR(err);
    if ((err = cudaFree(d_previous)) != cudaSuccess) ERR(err);
}