#include "Graph.h"
#include "../Macros.h"
#include <stdlib.h>

#ifndef _SHORTEST_H_
#define _SHORTEST_H_

#define INF 99999;
#define NEG_INF -99999;

void bellmanFord(graph_t* graph, int source, int* distance, int* previous);

int dijkstra_minVertex(int* distance, int* visited, int n);
void dijkstra_relax(int* distance, int* previous, int v, int u, int weight);

void dijkstra_graph(graph_t* graph, int source, int dest, int* distance, int* previous);
void dijkstra_map(int* map, int n, int m, int source, int dest, int* distance, int* previous);

#endif