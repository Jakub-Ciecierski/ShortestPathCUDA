extern "C"{
#include "graph.h"
}

#include "shortest_path_cuda.cuh"

__host__
int main()
{
    graph_t* graph = createGraph(5);
    addEdge(graph, 0, 3, 2);
    addEdge(graph, 0, 4, 1);
    addEdge(graph, 1, 2, 1);
    addEdge(graph, 2, 3, 3);
    addEdge(graph, 4, 2, 3);
    addEdge(graph, 3, 1, 2);

    showGraph(graph);

    int source = 0;
    int num_vertices = graph->num_vertices;
    int* graphMatrix = graph->adjMatrix;
    int* distance = (int*)malloc(num_vertices * sizeof(int));
    int* previous = (int*)malloc(num_vertices * sizeof(int));

    clock_t start, end;
    double delta;
    start = clock();

    shortestPathInit(graphMatrix, num_vertices, source, distance, previous);

    printf("\n********************** PRINTING RESULTS **********************\n\n");

    int i;
    for (i = 0; i < num_vertices; i++)
    {
        printf("Distance[%d] == %d \n", i, distance[i]);
    }
    printf("\n");
    for (i = 0; i < num_vertices; i++)
    {
        printf("Previous[%d] == %d \n", i, previous[i]);
    }

    end = clock();
    printf("Start: %d End: %d \n", start, end);
    delta = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Finished after %f \n", delta);

    return EXIT_SUCCESS;
}