#include "Tests/TestCases.h"

void usage(char* name)
{
    fprintf(stderr, "USAGE\n");
    fprintf(stderr, "%s: n m \n", name);
    fprintf(stderr, "OR \n");
    fprintf(stderr, "%s: test_id \n", name);
}

int main(int argc, char** argv)
{
    // custom dim: [n] x [m]
    if (argc == 3)
    {
        int n = atoi(argv[1]);
        int m = atoi(argv[2]);
        if (n <= 0 || m <= 0) 
        {
            usage(argv[0]);
            return EXIT_FAILURE;
        }
        test_case_custom(n,m);
    }
    // which test
    else if (argc == 2)
    {
        int which_test = atoi(argv[1]);

        switch (which_test)
        {
            case 0:
                test_case_very_small();
                break;
            case 1:
                test_case_small();
                break;
            case 2:
                test_case_medium();
                break;
            case 3:
                test_case_big();
                break;
            default:

                break;
        }
    }

    return EXIT_SUCCESS;
}