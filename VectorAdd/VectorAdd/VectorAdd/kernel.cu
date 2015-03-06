#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__
void vectorAddKernel(float* A, float* B, float* C, int n)
{
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    if (i < n)
    {
        C[i] = A[i] + B[i];
    }
}

__host__
int vectorAdd(float* A, float* B, float* C, int n)
{
    // size in bytes
    int size = n*sizeof(float);
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

    // x, y, z
    dim3 DimGrid((n - 1) / 256 + 1, 1, 1);
    dim3 DimBlock(256, 1, 1);
    
    printf("DimGrid: x: %d, y: %d, z: %d \n", DimGrid.x, DimGrid.y, DimGrid.z);
    printf("DimBlock: x: %d, y: %d, z: %d \n", DimBlock.x, DimBlock.y, DimBlock.z);

    printf("Initiating vectorAddKernel with %d elements \n", n);

    //call the kernel function
    vectorAddKernel <<<DimGrid, DimBlock >>>(d_A, d_B, d_C, n);

    // get the computed vector back to host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 1;
}

__host__
void vectorAddHost(float* A, float* B, float* C, int n)
{
    int i = 0;
    for (i = 0; i < n; i++)
    {
        C[i] = A[i] + B[i];
    }

}

void main()
{
    float *A, *B, *C;
    int n = 9999999;

    int size = n*sizeof(float);

    A = (float*)malloc(size);
    B = (float*)malloc(size);
    C = (float*)malloc(size);

    int i = 0;
    for (i = 0; i < n; i++)
    {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    clock_t start, end;
    double delta;
    start = clock();

    vectorAdd(A, B, C, n);

    end = clock();
    printf("Start: %d End: %d \n", start, end);
    delta = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Finished after %f \n", delta);

    for (i = 0; i < n; i++)
    {
        if (C[i] != 3.0f)
        {
            printf("Wrong calculations %d\n",i);
        }
    }

    free(A);
    free(B);
    free(C);
}
