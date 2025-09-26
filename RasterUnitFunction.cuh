#ifndef RASTER_FUNC
#define RASTER_FUNC

#include <cuda_runtime.h>

__device__ float atomicMinFloat(float* address, float value)
{
	unsigned int* address_as_ui = (unsigned int*)address;
	float ret = *address;
	while (value < ret)
	{
		float assume = ret;
		ret = __uint_as_float(atomicCAS(address_as_ui, __float_as_uint(assume), __float_as_uint(value)));
	}

	return ret;
}



#endif // !RASTER_FUNC
