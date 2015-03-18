#include "shortest_path.h"
#define INF 99999;
#define NEG_INF -99999;

void bellmanFord(graph_t* graph, int source, int* distance, int* previous)
{
    int num_vertices = graph->num_vertices;

    int i = 0;
    for (i = 0; i < num_vertices; i++)
    {
        distance[i] = INF;
        previous[i] = -1;
    }
    distance[source] = 0;

    for (i = 1; i < num_vertices; i++)
    {
        int u = 0;
        for (u = 0; u < num_vertices; u++)
        {
            int v = 0;
            for (v = 0; v < num_vertices; v++)
            {
                if (!areConnected(graph, u, v))
                    continue;
                // relax(u,v)
                if (distance[v] > distance[u] + getWeight(graph, u, v))
                {
                    distance[v] = distance[u] + getWeight(graph, u, v);
                    previous[v] = u;
                }
            }
        }
    }

    for (i = 1; i <= num_vertices ; i++)
    {
        int u = 0;
        for (u = 0; u < num_vertices; u++)
        {
            int v = 0;
            for (v = 0; v < num_vertices; v++)
            {
                if (!areConnected(graph, u, v))
                    continue;
                if (distance[v] > distance[u] + getWeight(graph, u, v))
                {
                    distance[v] = NEG_INF;
                    previous[v] = u;
                }
            }
        }
    }
}

int dij_minVertex(int* distance, int* visited, int n)
{
    int i = 0;
    int min = 0;
    // take first element that has not been visited yet
    for (i = 0; i < n; i++)
    {
        if (visited[i] == 0)
        {
            min = i;
            break;
        }
    }

    // find min
    for (i = 0; i < n; i++)
    {
        if ((distance[min] > distance[i]) && visited[i] == 0)
        {
            min = i;
        }
    }

    return min;
}

void dijkstra(graph_t* graph, int source, int dest, int* distance, int* previous)
{
    int num_vertices = graph->num_vertices;
    int* visited = (int*)malloc(num_vertices*sizeof(int)); // 0 unvisited, 1 visited
    
    int i = 0;
    for (i = 0; i < num_vertices; i++)
    {
        visited[i] = 0;
        distance[i] = INF;
        previous[i] = -1;
    }

    distance[source] = 0;

    int u = source;
    while (u != dest)
    {
        u = dij_minVertex(distance, visited, num_vertices);
        visited[u] = 1;

        int j = 0;
        neighborhood_t* nb = getNeighbors(graph, u);
        for (j = 0; j < nb->length; j++)
        {
            int v = nb->vertices[j];
            int dist = distance[u] + getWeight(graph, u, v);
            if (dist < distance[v])
            {
                distance[v] = dist;
                previous[v] = u;
            }
        }
    }
}