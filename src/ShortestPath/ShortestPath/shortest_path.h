#include "graph.h"

#ifndef _SHORTEST_H_
#define _SHORTEST_H_

void bellmanFord(graph_t* graph, int source, int* distance, int* previous);

void dijkstra(graph_t* graph, int source, int dest, int* distance, int* previous);
int dij_minVertex(int* distance, int* visited, int n);
#endif