#include "RasterParallelAlgorithm.h"
#include "RasterUnitFunction.cuh"
#include "RasterParallelAlgorithm.cuh"

namespace cg = cooperative_groups;

namespace
{
	unsigned int* dBlockSumBuffer = nullptr; // warp sum buffer used in inter-block scan, set to 1024 as default
	unsigned int* dSum = nullptr;

	constexpr int blockSize = 256;
}



// suggest block size 256 -> can compact ~ 500 thousands primitives
__global__ void PrimitiveCompactionKernel(int size, const Primitive* inputStream, Primitive* outputStream, unsigned int* gBlockScanBuffer , unsigned int* sum)
{
	__shared__ unsigned int sWarpScanBuffer[32];
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

	if (threadIdx.x < 32)
	{
		sWarpScanBuffer[threadIdx.x] = 0;
	}
	__syncthreads();

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
		if (threadIdx.x < 32) sWarpScanBuffer[threadIdx.x] = 0;
		__syncthreads();
		uint4 idata = ((uint4*)gBlockScanBuffer)[threadIdx.x];
		uint4 odata = ScanInBlock4Exclusive(idata, sWarpScanBuffer);
		((uint4*)gBlockScanBuffer)[threadIdx.x] = odata;
		__syncthreads();
		if (threadIdx.x == 0)
		{
			*sum = gBlockScanBuffer[1023];
		}
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
	CUDA_CHECK(cudaMalloc((void**)&dSum, sizeof(unsigned int)));
}

void  PrimitiveCompaction(int size, const Primitive* inputStream, Primitive* outputStream, unsigned int* sum, cudaStream_t stream)
{
	int numBlocks = (size + blockSize - 1) / blockSize;
	CUDA_CHECK(cudaMemsetAsync(dBlockSumBuffer, 0, sizeof(unsigned int) * 1024, stream));
	CUDA_CHECK(cudaMemsetAsync(dSum, 0, sizeof(unsigned int), stream));
	
	void* args[] = { &size, &inputStream, &outputStream, &dBlockSumBuffer, &dSum };
	cudaLaunchCooperativeKernel(PrimitiveCompactionKernel,numBlocks, blockSize, args, std::max(blockSize / 32u, 32u) * sizeof(unsigned int),stream);
	CUDA_CHECK(cudaMemcpyAsync(sum, dSum, sizeof(unsigned int), cudaMemcpyDeviceToDevice, stream));
}

void DestroyCompactionEnvironment()
{
	CUDA_CHECK(cudaFree(dBlockSumBuffer));
	CUDA_CHECK(cudaFree(dSum));
}