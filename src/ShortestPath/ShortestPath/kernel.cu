/*
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>


__host__
void matrixMultiplicationShared(float* A, float* B, float* C, int n)
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
    matrixMultiplicationKernelSharedMem << <DimGrid, DimBlock >> >(d_A, d_B, d_C, n);

    // get the computed vector back to host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

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

int main()
{
    float *A, *B, *C;
    int n = 64;
    int t = TILE_WIDTH;
    int size = n*n*sizeof(float);

    A = (float*)malloc(size);
    B = (float*)malloc(size);
    C = (float*)malloc(size);

    int i = 0;
    int j = 0;
    float value = 0.0f;
    for (i = 0; i < n*n; i++)
    {
        A[i] = value;
        B[i] = value;
        C[i] = 0.0f;
        value++;
    }

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            printf("A[%d,%d] = %lf\n", i, j, A[i*n + j]);
        }
    }

    clock_t start, end;
    double delta;
    start = clock();

    matrixMultiplication(A, B, C, n);
    for (i = 0; i < n*n; i++)
    {
        C[i] = 0.0f;
    }
    matrixMultiplicationShared(A, B, C, n);

    end = clock();
    printf("Start: %d End: %d \n", start, end);
    delta = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Finished after %f \n", delta);


    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            printf("C[%d,%d] = %lf\n", i, j, C[i*n + j]);
        }
    }

    free(A);
    free(B);
    free(C);
    return 0;
}
*/