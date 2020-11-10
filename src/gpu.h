#ifndef GPU_H
#define GPU_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BDIMX 16 // tile (and threadblock) size in x
#define BDIMY 16 // tile (and threadblock) size in y

//void gpuAssert(cudaError_t code, const char *file, int line);
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

#ifdef __cplusplus
}
#endif

#endif
