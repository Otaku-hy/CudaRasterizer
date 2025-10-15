#pragma once

#include <cuda_runtime.h>

inline __device__ unsigned int ScanInWarpInclusive(unsigned int laneVal)
{
	unsigned int lane_id = get_lane_id();
	unsigned int offset = 1;
	unsigned int otherLaneVal = __shfl_up_sync(0xFFFFFFFF, laneVal, offset, 32);
	laneVal += (lane_id >= offset) ? otherLaneVal : 0;
	offset *= 2;

	otherLaneVal = __shfl_up_sync(0xFFFFFFFF, laneVal, offset, 32);
	laneVal += (lane_id >= offset) ? otherLaneVal : 0;
	offset *= 2;

	otherLaneVal = __shfl_up_sync(0xFFFFFFFF, laneVal, offset, 32);
	laneVal += (lane_id >= offset) ? otherLaneVal : 0;
	offset *= 2;

	otherLaneVal = __shfl_up_sync(0xFFFFFFFF, laneVal, offset, 32);
	laneVal += (lane_id >= offset) ? otherLaneVal : 0;
	offset *= 2;

	otherLaneVal = __shfl_up_sync(0xFFFFFFFF, laneVal, offset, 32);
	laneVal += (lane_id >= offset) ? otherLaneVal : 0;

	return laneVal;
}

inline __device__ unsigned int ScanInWarpExclusive(unsigned int laneVal)
{
	unsigned int inclusive = ScanInWarpInclusive(laneVal);
	return inclusive - laneVal;
}

inline __device__ unsigned int ScanInBlockInclusive(unsigned int laneVal, unsigned int* sWarpSum)
{
	const unsigned numLane = 32u;

	unsigned int prefixInWarp = ScanInWarpInclusive(laneVal);
	unsigned int tidx = threadIdx.x + blockDim.x * threadIdx.y;

	if (get_lane_id() == numLane - 1)
	{
		unsigned int warpIndex = tidx / numLane;
		sWarpSum[warpIndex] = prefixInWarp;
	}
	__syncthreads();

	if (tidx < 32)
	{
		unsigned int warpSum = sWarpSum[tidx];
		unsigned int prefixOfWarp = ScanInWarpExclusive(warpSum);
		sWarpSum[tidx] = prefixOfWarp;
	}
	__syncthreads();

	return prefixInWarp + sWarpSum[tidx / 32];
}

inline __device__ unsigned int ScanInBlockExclusive(unsigned int laneVal, unsigned int* sWarpSum)
{
	unsigned int inclusive = ScanInBlockInclusive(laneVal, sWarpSum);
	return inclusive - laneVal;
}

inline __device__ uint4 ScanInBlock4Inclusive(uint4 idata, unsigned int* sWarpSum)
{
	idata.y += idata.x;
	idata.z += idata.y;
	idata.w += idata.z;

	unsigned int exclusive = ScanInBlockExclusive(idata.w, sWarpSum);
	uint4 odata;
	odata.x = idata.x + exclusive;
	odata.y = idata.y + exclusive;
	odata.z = idata.z + exclusive;
	odata.w = idata.w + exclusive;

	return odata;
}

inline __device__ uint4 ScanInBlock4Exclusive(uint4 idata, unsigned int* sWarpSum)
{
	uint4 inclusive = ScanInBlock4Inclusive(idata, sWarpSum);
	inclusive.x -= idata.x;
	inclusive.y -= idata.y;
	inclusive.z -= idata.z;
	inclusive.w -= idata.w;
	return inclusive;
}