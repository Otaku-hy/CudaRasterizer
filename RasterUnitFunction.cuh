#ifndef RASTER_FUNC
#define RASTER_FUNC

#include <cuda_runtime.h>

inline __device__ float atomicMinFloat(float* address, float value)
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

inline __device__ unsigned int get_lane_id()
{
	unsigned int ret;
	asm("mov.u32 %0, %laneid;" : "=r"(ret));
	return ret;
}
#endif // !RASTER_FUNC
