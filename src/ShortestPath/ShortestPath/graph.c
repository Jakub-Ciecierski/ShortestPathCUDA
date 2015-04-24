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
        graph->adjMatrix[i] = 0;
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
    graph->num_vertices++;
    int newSize = graph->num_vertices++ * graph->num_vertices++;
    graph->adjMatrix = (int*)realloc(graph->adjMatrix, sizeof(graph->adjMatrix) * newSize);
}

int getWeight(graph_t* graph, int u, int v)
{
    return graph->adjMatrix[graph->num_vertices * u + v];
}

int areConnected(graph_t* graph, int u, int v)
{
    return graph->adjMatrix[u * graph->num_vertices + v] != 0 ?
        1 : 0;
}
/*
neighborhood_t* getNeighbors(graph_t* graph, int u)
{
    neighborhood_t* nb = (neighborhood_t*)malloc(sizeof(neighborhood_t));
    nb->vertices = NULL;

    int n = graph->num_vertices;
    int length = 0;

    int i = 0;
    for (i = 0; i < n; i++)
    {
        if (areConnected(graph, u, i) == 1)
        {
            nb->vertices = realloc(nb->vertices, (++length)*sizeof(int));
            nb->vertices[length - 1] = i;
        }
    }

    nb->length = length;
    return nb;
}*/

void deleteGraph(graph_t* graph)
{
    free(graph->adjMatrix);
    free(graph);
}

void showGraph(graph_t* graph)
{
    int n = graph->num_vertices;
    int i = 0;
    
    for (i = 0; i<n; ++i)
    {
        int j = 0;
        printf("| ");
        for (j = 0; j<n; ++j)
        {
            printf("%d", graph->adjMatrix[i*n + j]);
            printf(" ");
        }
        printf("|\n");
    }
}

void err_exit(char* msg)
{
    printf("[Fatal Error]: %s \nExiting...\n", msg);
    exit(1);
}