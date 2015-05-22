#include "ShortestPath.h"

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

int dijkstra_minVertex(int* distance, int* visited, int n)
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

void dijkstra_relax(int* distance, int* previous, int v, int u, int weight)
{
    if (weight > 0)
    {
        int dist = distance[v] + weight;
        if (dist < distance[u])
        {
            distance[u] = dist;
            previous[u] = v;
        }
    }
}

void dijkstra_graph(graph_t* graph, int source, int dest, int* distance, int* previous)
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
        u = dijkstra_minVertex(distance, visited, num_vertices);
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

void dijkstra_map(int* map, int n, int m, int source, int dest, int* distance, int* previous)
{
    int i, j;
    int v, u;
    int row, col;
    int num_vertices = n*m;
    int* visited = (int*)malloc(num_vertices*sizeof(int)); // 0 unvisited, 1 visited

    for (i = 0; i < num_vertices; i++)
    {
        visited[i] = 0;
        distance[i] = INF;
        previous[i] = -1;
    }
    distance[source] = 0;

    v = source;
    while (v != dest)
    {
        v = dijkstra_minVertex(distance, visited, num_vertices);
        row = v / m;
        col = v % m;

        visited[v] = 1;

        //printf("\n\nRound# v:[%d] = row: %d, col: %d \n", v, row, col);

        int upper_i = row - 1;
        int upper_j = col;

        int right_i = row;
        int right_j = col + 1;

        int bottom_i = row + 1;
        int bottom_j = col;

        int left_i = row;
        int left_j = col - 1;

        if (upper_i >= 0) {
            u = upper_i*m + upper_j;
            //printf("NB# u:[%d] = row: %d, col: %d \n", u, upper_i, upper_j);
            int weight = map[u];
            dijkstra_relax(distance, previous, v, u, weight);
        }

        if (right_j <= m - 1){
            u = right_i*m + right_j;
            //printf("NB# u:[%d] = row: %d, col: %d \n", u, right_i, right_j);
            int weight = map[u];
            dijkstra_relax(distance, previous, v, u, weight);
        }

        if (bottom_i <= n - 1){
            u = bottom_i*m + bottom_j;
            //printf("NB# u:[%d] = row: %d, col: %d \n", u, bottom_i, bottom_j);
            int weight = map[u];
            dijkstra_relax(distance, previous, v, u, weight);
        }

        if (left_j >= 0){
            u = left_i*m + left_j;
            //printf("NB# u:[%d] = row: %d, col: %d \n", u, left_i, left_j);
            int weight = map[u];
            dijkstra_relax(distance, previous, v, u, weight);
        }
    }
}
