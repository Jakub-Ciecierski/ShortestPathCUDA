#ifndef _GRAPH_H_
#define _GRAPH_H_

typedef struct graph
{
    int num_vertices;
    int num_edges;
    int* adjMatrix;
}graph_t;

graph_t* createGraph(int n);

void addEdge(graph_t* graph, int u, int v, int weight);

void addNode(graph_t* graph);

int getWeight(graph_t* graph, int u, int v);

void deleteGraph(graph_t* graph);

void err_exit(char* msg);

#endif