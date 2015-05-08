#define MAX_GRID_DIM_SIZE 65535

#define MAX_BLOCK_THREAD_COUNT 512

#define MAX_BLOCK_DIM_SIZE 512

#define INF 99999

#define ERR(err) (printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__),\
                    exit(EXIT_FAILURE))