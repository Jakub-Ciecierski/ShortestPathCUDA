#include "graph.h"
#include <stdlib.h>

graph_t* createGraph(int n)
{
    graph_t* graph = (graph_t*)malloc(sizeof(graph_t));
    if (!graph)
        err_exit("createGraph, malloc()");

    graph->adjMatrix = (int*)malloc(n*n * sizeof(int));
    if (!graph->adjMatrix)
        err_exit("createGraph, malloc()");

    int i = 0;
    for (i = 0; i < n*n; i++)
    {
        graph->adjMatrix = 0;
    }

    graph->num_vertices = n;
    graph->num_edges = 0;

    return graph;
}

void addEdge(graph_t* graph, int u, int v, int weight)
{
    graph->adjMatrix[graph->num_vertices * u + v] = weight;
    graph->num_edges++;
}

void addNode(graph_t* graph)
{
    graph->adjMatrix = (int*)realloc(graph->adjMatrix, sizeof(graph->adjMatrix) + (graph->num_vertices*2 + 1));
    graph->num_vertices++;
}

int getWeight(graph_t* graph, int u, int v)
{
    return graph->adjMatrix[graph->num_vertices * u + v];
}

void deleteGraph(graph_t* graph)
{
    free(graph->adjMatrix);
    free(graph);
}

void err_exit(char* msg)
{
    printf("[Fatal Error]: %s \nExiting...\n", msg);
    exit(1);
}