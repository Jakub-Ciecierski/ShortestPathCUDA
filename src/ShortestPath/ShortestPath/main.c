#include <stdio.h>
#include "graph.h"
#include "shortest_path.h"

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

    int num_vertices = graph->num_vertices;
    int* distance = (int*)malloc(num_vertices * sizeof(int));
    int* previous = (int*)malloc(num_vertices * sizeof(int));

    //bellmanFord(graph, 0, distance, previous);
    dijkstra(graph, 0, 2, distance, previous);

    int i = 0;
    for (i = 0; i < num_vertices; i++)
    {
        printf("Distance[%d] == %d \n", i, distance[i]);
    }

    for (i = 0; i < num_vertices; i++)
    {
        printf("Previous[%d] == %d \n", i, previous[i]);
    }
    
    return 0;
}
