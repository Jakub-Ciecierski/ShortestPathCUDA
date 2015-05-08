extern "C"{
#include "graph.h"
}
extern "C"{
#include "shortest_path.h"
}

#include <math.h>

#include "shortest_path_cuda.cuh"

#define DIR "benchmarks"

void test_case_small();
void test_case_small2();
void test_case_medium();
void test_case_big();

void safeMapToFile(int* map, int* path, int n, int m, char* filename);

__host__
int main()
{
    //test_case_small();

    //test_case_small2();

    //test_case_medium();

    test_case_big();

    return EXIT_SUCCESS;
}

__host__
void test_case_small()
{
    const int n = 3;
    const int m = 3;
    const int vertex_count = n*m;

    int map[vertex_count] = { 1, 1, 15,
                            1, 100, 2,
                            15, 0, 1};

    graph_t* graph = mapToGraph(map, n, m);

    int source = 0;
    int dest = vertex_count-1;
    int num_vertices = graph->num_vertices;
    int* graphMatrix = graph->adjMatrix;
    int* distance = (int*)malloc(num_vertices * sizeof(int));
    int* previous = (int*)malloc(num_vertices * sizeof(int));
    int* shortest_path = NULL;

    clock_t start, end;
    double delta;
    start = clock();

    // device
    shortestPathInit(graphMatrix, num_vertices, source, dest, distance, previous);
    shortest_path = getShortestPath(previous, vertex_count, source, dest);
    safeMapToFile(map, shortest_path, n, m, "../../../benchmarks/map_small_device.txt");
    free(shortest_path);

    end = clock();
    delta = ((double)(end - start)) / CLOCKS_PER_SEC;

    // host
    dijkstra(graph, source, dest, distance, previous);
    shortest_path = getShortestPath(previous, vertex_count, source, dest);
    safeMapToFile(map, shortest_path, n, m, "../../../benchmarks/map_small_host.txt");
    free(shortest_path);
}

__host__
void test_case_small2()
{
    const int n = 7;
    const int m = 7;
    const int vertex_count = n*m;

    int map[vertex_count] = {10, 10, 10, 10, 10, 10, 10,
                            999, 999, 999, 999, 999, 999, 10,
                            999, 999, 10, 10, 10, 10, 10,
                            10, 10, 10, 0, 999, 999, 999,
                            10, 10, 0, 10, 10, 10, 10,
                            10, 10, 10, 10, 999, 999, 10,
                            10, 10, 0, 0, 0, 999, 10 };

    graph_t* graph = mapToGraph(map, n, m);

    int source = 0;
    int dest = vertex_count-1;
    int num_vertices = graph->num_vertices;
    int* graphMatrix = graph->adjMatrix;
    int* distance = (int*)malloc(num_vertices * sizeof(int));
    int* previous = (int*)malloc(num_vertices * sizeof(int));
    int* shortest_path = NULL;

    clock_t start, end;
    double delta;
    start = clock();

    // device
    shortestPathInit(graphMatrix, num_vertices, source, dest, distance, previous);
    shortest_path = getShortestPath(previous, vertex_count, source, dest);
    safeMapToFile(map, shortest_path, n, m, "../../../benchmarks/map_small_device.txt");
    free(shortest_path);

    end = clock();
    delta = ((double)(end - start)) / CLOCKS_PER_SEC;

    // host
    dijkstra(graph, source, dest, distance, previous);
    shortest_path = getShortestPath(previous, vertex_count, source, dest);
    safeMapToFile(map, shortest_path, n, m, "../../../benchmarks/map_small_host.txt");
    free(shortest_path);
}

__host__
void test_case_medium()
{
    const int n = 100;
    const int m = 100;
    const int vertex_count = n*m;

    srand(NULL);
    int w;

    int i, j;
    int map[vertex_count];
    for (i = 0; i < n; i++){
        for (j = 0; j < m; j++){
            w = rand() % 10;
            map[i * n + j] = w;
        }
    }
    graph_t* graph = mapToGraph(map, n, m);
    
    int source = 0;
    int dest = vertex_count-1;
    int num_vertices = graph->num_vertices;
    int* graphMatrix = graph->adjMatrix;
    int* distance = (int*)malloc(num_vertices * sizeof(int));
    int* previous = (int*)malloc(num_vertices * sizeof(int));
    int* shortest_path = NULL;

    clock_t start, end;
    double delta;
    start = clock();

    // device
    shortestPathInit(graphMatrix, num_vertices, source, dest, distance, previous);
    shortest_path = getShortestPath(previous, vertex_count, source, dest);
    safeMapToFile(map, shortest_path, n, m, "../../../benchmarks/map_medium_device.txt");
    free(shortest_path);

    end = clock();
    delta = (((float)end - (float)start)) / CLOCKS_PER_SEC * 1000;
    printf("%d ms\n", delta);

    // host
    dijkstra(graph, source, dest, distance, previous);
    shortest_path = getShortestPath(previous, vertex_count, source, dest);
    safeMapToFile(map, shortest_path, n, m, "../../../benchmarks/map_medium_host.txt");
    free(shortest_path);
}

__host__
void test_case_big()
{
    //const int _n = (int)sqrt((double)INT_MAX);
    const int n = 20000;
    const int m = 20000;
    const int vertex_count = n*m;
    
    srand(NULL);
    int w;

    int i, j;
    int map[vertex_count];
    for (i = 0; i < n; i++){
        for (j = 0; j < m; j++){
            w = rand() % 10;
            map[i * n + j] = w;
        }
    }
    graph_t* graph = mapToGraph(map, n, m);

    int source = 0;
    int dest = vertex_count - 1;
    int num_vertices = graph->num_vertices;
    int* graphMatrix = graph->adjMatrix;
    int* distance = (int*)malloc(num_vertices * sizeof(int));
    int* previous = (int*)malloc(num_vertices * sizeof(int));
    int* shortest_path = NULL;

    clock_t start, end;
    double delta;
    start = clock();

    // device
    shortestPathInit(graphMatrix, num_vertices, source, dest, distance, previous);
    shortest_path = getShortestPath(previous, vertex_count, source, dest);
    //safeMapToFile(map, shortest_path, n, m, "../../../benchmarks/map_medium_device.txt");
    free(shortest_path);

    end = clock();
    delta = (((float)end - (float)start)) / CLOCKS_PER_SEC * 1000;
    /*
    // host
    dijkstra(graph, source, dest, distance, previous);
    shortest_path = getShortestPath(previous, vertex_count, source, dest);
    safeMapToFile(map, shortest_path, n, m, "../../../benchmarks/map_medium_host.txt");
    free(shortest_path);
    */
}

__host__
void safeMapToFile(int* map, int* shortest_path, int n, int m, char* filename)
{
    FILE* f = fopen(filename, "w");

    fprintf(f, "MAP\nDIM %d %d\n", n, m);

    int i, j;
    for (i = 0; i < n; i++){
        for (j = 0; j < m; j++){
            fprintf(f, "%d ", map[i*n+j]);
        }
        fprintf(f, "\n");
    }

    if (shortest_path != NULL){
        fprintf(f, "PATH\n");
        for (i = 0; i < n*m; i++){
            if (shortest_path[i] != -1)
                fprintf(f, "%d\n", shortest_path[i]);
        }
    }
    fprintf(f, "EOF\n");
    fclose(f);
}