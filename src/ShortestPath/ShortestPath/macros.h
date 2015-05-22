#ifndef _MACROS_H_
#define _MACROS_H_

#define ERR(source) (fprintf(stderr,"%s:%d\n",__FILE__,__LINE__),\
                     perror(source),\
		     		     exit(EXIT_FAILURE))
#endif