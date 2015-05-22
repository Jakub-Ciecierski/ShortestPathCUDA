#include "TestCases.h"

void test_case_host_graph(int* map, int n, int m, int source, int dest, char* file_path)
{
    /******************** DATA SETS ************************/
    int vertex_count = n*m;

    int* distance = (int*)malloc(vertex_count  * sizeof(int));
    if (!distance) ERR("distance malloc()");

    int* previous = (int*)malloc(vertex_count  * sizeof(int));
    if (!previous) ERR("previous malloc()");

    int* shortest_path = NULL;

    /******************** ALGORITHM ************************/

    fprintf(stdout, "\n\n********************************** HOST GRAPH **********************************\n\n");

    graph_t* graph = mapToGraph(map, n, m);
    int num_vertices = graph->num_vertices;
    int* graphMatrix = graph->adjMatrix;

    float hostTime;
    double startTime, stopTime, elapsed;
    startTime = second();

    // host algorithm
    dijkstra_graph(graph, source, dest, distance, previous);

    stopTime = second();
    hostTime = (stopTime - startTime) * 1000;

    fprintf(stdout, "Host_graph finished after: %f ms \n", hostTime);

    shortest_path = getShortestPath(previous, vertex_count, source, dest);
    safeResults(file_path, map, shortest_path, n, m, CPU, hostTime);

    free(distance);
    free(previous);
    free(shortest_path);
}

void test_case_host_map(int* map, int n, int m, int source, int dest, char* file_path)
{
    /******************** DATA SETS ************************/

    int vertex_count = n*m;

    int* distance = (int*)malloc(vertex_count  * sizeof(int));
    if (!distance) ERR("distance malloc()");

    int* previous = (int*)malloc(vertex_count  * sizeof(int));
    if (!previous) ERR("previous malloc()");

    int* shortest_path = NULL;

    /******************** ALGORITHM ************************/

    fprintf(stdout, "\n\n********************************** HOST MAP **********************************\n\n");

    float hostTime;
    double startTime, stopTime, elapsed;
    startTime = second();

    // host algorithm
    dijkstra_map(map, n, m, source, dest, distance, previous);

    stopTime = second();
    hostTime = (stopTime - startTime) * 1000;

    fprintf(stdout, "Host_map finished after: %f ms \n", hostTime);

    shortest_path = getShortestPath(previous, vertex_count, source, dest);
    safeResults(file_path, map, shortest_path, n, m, CPU, hostTime);

    free(distance);
    free(previous);
    free(shortest_path);
}

void test_case_device(int* map, int n, int m, int source, int dest, char* file_path)
{
    /******************** DATA SETS ************************/

    int vertex_count = n*m;

    int* distance = (int*)malloc(vertex_count  * sizeof(int));
    if (!distance) ERR("distance malloc()");

    int* previous = (int*)malloc(vertex_count  * sizeof(int));
    if (!previous) ERR("previous malloc()");

    int* shortest_path = NULL;

    /******************** ALGORITHM ************************/

    fprintf(stdout, "\n\n********************************** DEVICE **********************************\n\n");

    float kernelTime;
    float deviceTotalTime;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // device algorithm
    shortestPathInit(map, vertex_count, n, m, source, dest, distance, previous, &kernelTime);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&deviceTotalTime, start, stop);

    shortest_path = getShortestPath(previous, vertex_count, source, dest);
    safeResults(file_path, map, shortest_path, n, m, GPU, deviceTotalTime, kernelTime);

    free(distance);
    free(previous);
    free(shortest_path);
}

void test_case_logic(int* map, int n, int m, char* device_file, char* host_file)
{
    time_t t = time(NULL);
    struct tm tm = *localtime(&t);

    char time[FILE_SIZE];
    if (sprintf_s(time, FILE_SIZE, "%d-%d-%d_%d;%d;%d", tm.tm_mday, tm.tm_mon + 1, tm.tm_year + 1900,
                                                        tm.tm_hour, tm.tm_min, 
                                                        tm.tm_sec) < 0) ERR("sprintf_s");

    char device_path[FILE_SIZE];
    if (sprintf_s(device_path, FILE_SIZE, "%s%s_%s%s", DIR, device_file, time, FILE_EXTENSION) < 0) ERR("sprintf_s");

    char host_map_path[FILE_SIZE];
    if (sprintf_s(host_map_path, FILE_SIZE, "%s%s_map_%s%s", DIR, host_file, time, FILE_EXTENSION) < 0) ERR("sprintf_s");

    char host_graph_path[FILE_SIZE];
    if (sprintf_s(host_graph_path, FILE_SIZE, "%s%s_graph_%s%s", DIR, host_file, time, FILE_EXTENSION) < 0) ERR("sprintf_s");

    int vertex_count = n*m;

    int source = 0;
    int dest = vertex_count - 1;
    while (map[dest] == 0){
        dest--;
    }

    fprintf(stdout, "Finding path for: source[%d], destination[%d] \n", source, dest);

    test_case_device(map, n, m, source, dest, device_path);

    //test_case_host_map(map, n, m, source, dest, host_map_path);
    
    //test_case_host_graph(map, n, m, source, dest, host_graph_path);

    //free(map);
}

void test_case_very_small()
{
    /*
    #define N 3
    #define M 3
    #define VERTEX_COUNT M*N
    
    int map[VERTEX_COUNT] = { 1, 1, 15,
                            1, 100, 2,
                            15, 0, 1 };
                            */
    #define N 6
    #define M 3
    #define VERTEX_COUNT M*N

    int map[VERTEX_COUNT] = {   1, 1, 15,
                                100, 100, 2,
                                15, 0, 1 ,
                                2, 1, 2,
                                1,35,35,
                                1, 1, 15, };
    
    test_case_logic(map, N, M,
        "very_small_device",
        "very_small_host");
}

void test_case_small()
{
    #define N 7
    #define M 7
    #define VERTEX_COUNT M*N

    int map[VERTEX_COUNT] = { 10, 10, 10, 10, 10, 10, 10,
                            999, 999, 999, 999, 999, 999, 10,
                            999, 999, 10, 10, 10, 10, 10,
                            10, 10, 10, 0, 999, 999, 999,
                            10, 10, 0, 10, 10, 10, 10,
                            10, 10, 10, 10, 999, 999, 10,
                            10, 10, 0, 0, 0, 999, 10 };

    test_case_logic(map, N, M,
        "small_device",
        "small_host");
}

void test_case_medium()
{
    #define N 100
    #define M 100
    #define VERTEX_COUNT M*N

    srand(NULL);

    int i, j;
    int map[VERTEX_COUNT];
    for (i = 0; i < N; i++){
        for (j = 0; j < M; j++){
            map[i * N + j] = rand() % 10;
        }
    }
    test_case_logic(map, N, M,
        "medium_device",
        "medium_host");
}

void test_case_big()
{
    #define N 2000
    #define M 2000
    #define VERTEX_COUNT M*N

    srand(NULL);

    int i, j;
    int* map = (int*)malloc(VERTEX_COUNT*sizeof(int));
    if (!map) ERR("map malloc");

    for (i = 0; i < N; i++){
        for (j = 0; j < M; j++){
            map[i * N + j] = rand() % 10;
        }
    }
    test_case_logic(map, N, M,
        "big_device",
        "big_host");
}

void test_case_custom(int n, int m)
{
    int vertex_count = n*m;

    fprintf(stdout, "\nMap Config: \n");
    fprintf(stdout, "DIM [%d] x [%d] \n", n, m);
    fprintf(stdout, "VERTEX COUNT: %d \n\n",vertex_count);
    
    srand(NULL);

    int i, j;
    int* map = (int*)malloc(vertex_count*sizeof(int));
    if (!map) ERR("map malloc");

    fprintf(stdout, " >> Initilizing map \n");
    for (i = 0; i < n; i++){
        for (j = 0; j < m; j++){
            map[i * m + j] = rand() % 10;
        }
    }
    fprintf(stdout, " >> Finished Initilizing map \n");

    test_case_logic(map, n, m,
        "custom_device",
        "custom_host");
}

/////////////////////
/// Example of Structure of the file:
///
/// TYPE
/// CPU
///
/// RESULTS
/// 89.01223
///
/// MAP 
/// DIM 3 3
/// 0 0 2 
/// 2 1 4
/// 9 1 2
///
/// PATH
/// 0
/// 1
/// 5
/// 8
/// EOF
/////////////////////
void safeResults(char* filename, int* map, int* shortest_path, int n, int m,
                int type, float totalTime, float kernelTime)
{
    int i, j;
    FILE* f = fopen(filename, "w");
    if (f == NULL) ERR("fopen()");

    fprintf(f, "TYPE\n");
    if (type == CPU){
        fprintf(f, "CPU\n");
    }if (type == GPU){
        fprintf(f, "GPU\n");
    }

    fprintf(f, "\n");

    fprintf(f, "RESULTS\n");
    fprintf(f, "TOTAL TIME\n");
    fprintf(f, "%f\n", totalTime);
    if (type == GPU){
        fprintf(f, "KERNEL TIME\n");
        fprintf(f, "%f\n", kernelTime);
    }

    fprintf(f, "\n");

    if (map != NULL){
        fprintf(f, "MAP\nDIM %d %d\n", n, m);
        for (i = 0; i < n; i++){
            for (j = 0; j < m; j++){
                fprintf(f, "%d ", map[i*m + j]);
            }
            fprintf(f, "\n");
        }
    }

    fprintf(f, "\n");

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