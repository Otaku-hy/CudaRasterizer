#include "RasterParallelAlgorithm.h"
#include "RasterUnitFunction.cuh"

namespace cg = cooperative_groups;

namespace
{
	unsigned int* dBlockSumBuffer; // warp sum buffer used in inter-block scan, set to 1024 as default

	constexpr int blockSize = 256;
}

__device__ unsigned int ScanInWarpInclusive(unsigned int laneVal)
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

__device__ unsigned int ScanInWarpExclusive(unsigned int laneVal)
{
	unsigned int inclusive = ScanInWarpInclusive(laneVal);
	return inclusive - laneVal;
}

__device__ unsigned int ScanInBlockInclusive(unsigned int laneVal, unsigned int* sWarpSum)
{
	unsigned int prefixInWarp = ScanInWarpInclusive(laneVal);
	if (get_lane_id() == 31)
	{
		unsigned int warpIndex = threadIdx.x / 32;
		sWarpSum[warpIndex] = prefixInWarp;
	}
	__syncthreads();

	if (threadIdx.x < 32)
	{
		unsigned int warpSum = sWarpSum[threadIdx.x];
		unsigned int prefixOfWarp = ScanInWarpExclusive(warpSum);
		sWarpSum[threadIdx.x] = prefixOfWarp;
	}
	__syncthreads();

	return prefixInWarp + sWarpSum[threadIdx.x / 32];
}

__device__ unsigned int ScanInBlockExclusive(unsigned int laneVal, unsigned int* sWarpSum)
{
	unsigned int inclusive = ScanInBlockInclusive(laneVal, sWarpSum);
	return inclusive - laneVal;
}

__device__ uint4 ScanInBlock4Inclusive(uint4 idata, unsigned int* sWarpSum)
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

__device__ uint4 ScanInBlock4Exclusive(uint4 idata, unsigned int* sWarpSum)
{
	uint4 inclusive = ScanInBlock4Inclusive(idata, sWarpSum);
	inclusive.x -= idata.x;
	inclusive.y -= idata.y;
	inclusive.z -= idata.z;
	inclusive.w -= idata.w;
	return inclusive;
}

// suggest block size 256 -> can compact ~ 500 thousands primitives
__global__ void PrimitiveCompactionKernel(int size, const Primitive* inputStream, Primitive* outputStream, unsigned int* gBlockScanBuffer , unsigned int* sum)
{
	extern __shared__ unsigned int sWarpScanBuffer[];
	const unsigned int leaderLane = 0;

	cg::grid_group grid = cg::this_grid();

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size && blockIdx.x != 0) return;

	bool predicate = idx < size ? (inputStream[idx].primitiveID != -1) : false;
	unsigned int ballotMask = __ballot_sync(0xFFFFFFFF, predicate);
	
	unsigned int  lane_id = get_lane_id();
	unsigned int lowerBits = ballotMask & ((1U << lane_id) - 1);
	unsigned int offsetInWarp = __popc(ballotMask & lowerBits );

	unsigned int totalValidInWarp = __popc(ballotMask);

	if (lane_id == leaderLane)
	{
		unsigned int warpIndex = threadIdx.x / 32;
		sWarpScanBuffer[warpIndex] = totalValidInWarp;
	}
	__syncthreads();

	if ( threadIdx.x < 32 ) //first warp
	{	
		unsigned int warpSum = sWarpScanBuffer[threadIdx.x];
		unsigned int warpPrefixSum = ScanInWarpExclusive(warpSum);
		sWarpScanBuffer[threadIdx.x] = warpPrefixSum;
		if (threadIdx.x == (blockDim.x + 31) / 32) gBlockScanBuffer[blockIdx.x] = warpPrefixSum + warpSum;
	}
	__syncthreads();
	unsigned int offsetInBlock = sWarpScanBuffer[threadIdx.x / 32] + offsetInWarp;

	grid.sync();

	if(blockIdx.x == 0) // the second pass only use the first block with every thread read uint4 vector to reduce
	{
		uint4 idata = ((uint4*)gBlockScanBuffer)[threadIdx.x];
		uint4 odata = ScanInBlock4Exclusive(idata, sWarpScanBuffer);
		((uint4*)gBlockScanBuffer)[threadIdx.x] = odata;
		*sum = gBlockScanBuffer[1023];
	}
	__syncthreads();

	grid.sync();

	if (predicate)
	{
		unsigned int blockPrefix = gBlockScanBuffer[blockIdx.x];
		unsigned int outputIndex = blockPrefix + offsetInBlock;
		outputStream[outputIndex] = inputStream[idx];
	}
}

void InitCompactionEnvironment()
{
	cudaMalloc((void**)&dBlockSumBuffer, sizeof(unsigned int) * 1024);
}

int  PrimitiveCompaction(int size, const Primitive* inputStream, Primitive* outputStream)
{
	int numBlocks = (size + blockSize - 1) / blockSize;
	CUDA_CHECK(cudaMemset(dBlockSumBuffer, 0, sizeof(unsigned int) * 1024));

	unsigned int* dSum = nullptr;
	unsigned int sum;
	CUDA_CHECK(cudaMalloc((void**)&dSum, sizeof(unsigned int)));

	void* args[] = { &size, &inputStream, &outputStream, &dBlockSumBuffer, &dSum };
	cudaLaunchCooperativeKernel(PrimitiveCompactionKernel,numBlocks, blockSize, args, std::max(blockSize / 32u, 32u) * sizeof(unsigned int),0);

	//PrimitiveCompactionKernel << <numBlocks, blockSize, c >> > (size, inputStream, outputStream, dBlockSumBuffer, dSum);
	CUDA_CHECK(cudaGetLastError());

	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_CHECK(cudaMemcpy(&sum, dSum, sizeof(unsigned int), cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

	cudaFree(dSum);

	return sum;
}

void DestroyCompactionEnvironment()
{
	cudaFree(dBlockSumBuffer);
}