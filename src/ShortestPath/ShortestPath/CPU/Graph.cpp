#include "Graph.h"

graph_t* createGraph(int n)
{
    graph_t* graph = (graph_t*)malloc(sizeof(graph_t));
    if (!graph) ERR("createGraph, malloc()");


    graph->adjMatrix = (int*)malloc(n*n * sizeof(int));
    if (!graph->adjMatrix) ERR("createGraph malloc()");

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
            nb->vertices = (int*)realloc(nb->vertices, (++length)*sizeof(int));
            nb->vertices[length - 1] = i;
        }
    }

    nb->length = length;
    return nb;
}

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

////////////////////
/// two dimensional map - n x n
/// Map based on Neumann neightborhood
////////////////////
graph_t* mapToGraph(int* map, int n, int m)
{
    int i;
    int j;
    graph_t* graph = createGraph(n*m);

    // for each cell in the map, create a its connection in the graph
    // maximum 4 connections per cell
    for (i = 0; i < n; i++) // row
    {
        for (j = 0; j < m; j++) // columns
        {
            int upper_i = i-1;
            int upper_j = j;

            int right_i = i;
            int right_j = j + 1;

            int bottom_i = i + 1;
            int bottom_j = j;

            int left_i = i;
            int left_j = j - 1;

            if (upper_i >= 0) {
                addEdge(graph, i*n + j, upper_i*n + upper_j, map[upper_i*n + upper_j]);
            }
                
            if (right_j <= m - 1){
                addEdge(graph, i*n + j, right_i*n + right_j, map[right_i*n + right_j]);
            }
                
            if (bottom_i <= n - 1){
                addEdge(graph, i*n + j, bottom_i*n + bottom_j, map[bottom_i*n + bottom_j]);
            }

            if (left_j >= 0){
                addEdge(graph, i*n + j, left_i*n + left_j, map[left_i*n + left_j]);
            }
                
        }
    }
    return graph;
}

int* getShortestPath(int* previous, int n, int source, int dest)
{
    int i;
    int prev_node = previous[dest];
    int* shortest_path = (int*)malloc(n * sizeof(int));
    if (shortest_path == NULL) ERR("shortest_path malloc");

    for (i = 0; i < n; i++)
        shortest_path[i] = -1;

    shortest_path[0] = dest;

    for (i = 1; i < n; i++){
        if (prev_node < 0) {
            fprintf(stderr, " << Path does not exist \n");
            return shortest_path;
        }
        shortest_path[i] = prev_node;
        prev_node = previous[prev_node];

        if (prev_node == source)
            break;
    }
    return shortest_path;
}