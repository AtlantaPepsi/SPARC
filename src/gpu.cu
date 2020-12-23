#include "gpu.h"


#ifdef __cplusplus
extern "C" {
#endif

void gpuAssert(cudaError_t code, const char *file, int line)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (1) exit(code);
   }
}

#ifdef __cplusplus
}
#endif
