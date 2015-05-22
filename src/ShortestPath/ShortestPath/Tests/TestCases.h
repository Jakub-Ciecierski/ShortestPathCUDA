#ifndef _TESTS_H_
#define _TESTS_H_

#include "../Macros.h"
#include "../CPU/Graph.h"
#include "../CPU/ShortestPath.h"
#include "../GPU/ShortestPathMap.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CPU 1
#define GPU 2

#define FILE_SIZE 512

#define FILE_EXTENSION ".txt"
#define DIR "../../../benchmarks/"

/******************************************************************/
/*********** The cross platfrom time measure for CPU **************/
/******************************************************************/

#if defined(_WIN32)
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
static double second(void)
{
    LARGE_INTEGER t;
    static double oofreq;
    static int checkedForHighResTimer;
    static BOOL hasHighResTimer;

    if (!checkedForHighResTimer) {
        hasHighResTimer = QueryPerformanceFrequency(&t);
        oofreq = 1.0 / (double)t.QuadPart;
        checkedForHighResTimer = 1;
    }
    if (hasHighResTimer) {
        QueryPerformanceCounter(&t);
        return (double)t.QuadPart * oofreq;
    }
    else {
        return (double)GetTickCount() / 1000.0;
    }
}
#elif defined(__linux__) || defined(__APPLE__)
#include <stddef.h>
#include <sys/time.h>
static double second(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}
#else
#error unsupported platform
#endif

/******************************************************************/
/************************** TEST CASES ****************************/
/******************************************************************/

void test_case_host_graph(int* map, int n, int m, int source, int dest, char* file_path);
void test_case_host_map(int* map, int n, int m, int source, int dest, char* file_path);
void test_case_device(int* map, int n, int m, int source, int dest, char* file_path);

void test_case_logic(int* map, int n, int m, char* device_file, char* host_file);

void test_case_very_small();
void test_case_small();
void test_case_medium();
void test_case_big();
void test_case_custom(int n, int m);

void safeResults(char* filename, int* map, int* shortest_path, int n, int m,
    int type, float totalTime, float kernelTime = 0);


#endif