#ifndef RASTER_FUNC
#define RASTER_FUNC

#include <cuda_runtime.h>

__device__ float atomicMinFloat(float* address, float value)
{
	float old = *address, assumed;
	unsigned int* address_as_ui = (unsigned int*)address;

	do {
		assumed = old;
		old = __uint_as_float(atomicCAS(address_as_ui, __float_as_uint(assumed), __float_as_uint(value)));
	} while (old > value);

	return old;
}



#endif // !RASTER_FUNC
