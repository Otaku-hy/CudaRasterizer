#pragma once

#include "Rasterizer.h"
#include "RasterUnitFunction.cuh"
#include "RasterParallelAlgorithm.cuh"

namespace CRPipeline
{
	extern const VertexVSIn* dInVertexStream;
	extern VertexVSOut* dOutVertexStream;
	extern const uint32_t* dIndexStream;
	extern Primitive* dPrimitiveStream;
	extern FragmentPSin* dFragmentStream;
	extern FragmentPSOut* dFragmentOutStream;
	extern unsigned* dDepthBuffer;
	extern float4* dRenderTarget;
	extern unsigned char* dFrameBuffer;
	extern unsigned int* dHiZ;

	extern int windowWidth;
	extern int windowHeight;

	extern int dcVertexCount;
	extern int dcIndexCount;
	extern int dcIndexCountPerPrimitive;
	extern int dcPrimitiveCount;

	extern TriangleSetupData* dTriSetupData;
	extern Primitive* dCompactedPrimitiveStream;
	extern int* dPixelBaseIdx;

	extern unsigned int* dSubTriangleCounter;
	extern unsigned int* dChunkAllocator;
	extern unsigned int* dTileChunkAllocator;
	extern unsigned int* dQuadAllocator;
	extern unsigned int* dBinSubQueueCounter;
	extern unsigned int* dPrimitiveCounter;

	extern unsigned int* hTileChunkCount;
	extern unsigned int* hBinChunkCount;
	extern unsigned int* hFragmentCount;
	extern unsigned int* hdcPrimitiveCount;

	extern unsigned int* dSubQueueBaseIndex;
	extern unsigned int* dSubQueuePrimCount;
	extern unsigned int* dBinQueue;
	extern unsigned int* dTileQueueBaseIndex;
	extern unsigned int* dTileQueuePrimCount;
	extern unsigned int* dTileQueue;

	extern void SetGraphicsRoot(unsigned char* outRenderTarget, unsigned* depthBuffer,
		const VertexVSIn* vertexStream, const uint32_t* indexStream,
		int indexCount, int vertexCount, MatricesCBuffer* cb, cudaStream_t stream);

	extern __global__ void VertexFetchAndShading(int dimension, const VertexVSIn* inVertexStream, VertexVSOut* outVertexStream);

	extern __global__ void PrimitiveAssembly(int dimension, const uint32_t* indexStream, const VertexVSOut* vertexStream, Primitive* primitiveStream, int width, int height, unsigned int* subTriAllocationCounter);

	extern __global__ void PrimitiveBinning(const unsigned int* pSize, const Primitive* primitiveStream, unsigned int* chunkAllocator,
		unsigned int* subQueueBaseIndex, unsigned int* subQueuePrimCount, unsigned int* queue, int width, int height);

	extern __global__ void TriangleSetup(const unsigned int* pSize, const Primitive* primitiveStream, TriangleSetupData* triPreparedStream);

	extern __global__ void CoarseRasterizer(const unsigned int* primitiveCount /*input a compacted primitive Size*/, const unsigned int* subQueueBaseIndex,
		const unsigned int* subQueuePrimCount, const unsigned* binQueue,
		const TriangleSetupData* triSetupData, unsigned* hiZ,
		unsigned* tileChunkAllocator, unsigned* tileQueueBaseIndex, unsigned* tileQueuePrimCount, unsigned int* tileQueue,
		int width, int height);

	extern __global__ void FineRasterizerWIP(const unsigned int* tileQueueBaseIndex, const unsigned int* tileQueuePrimCount, const unsigned int* tileQueue,
		const TriangleSetupData* triSetupData, unsigned* depthBuffer, unsigned* quadAllocator, FragmentPSin* fragmentStream, int width, int height);

	extern __global__ void PixelShader(unsigned int* dimension, const FragmentPSin* fragmentStream, FragmentPSOut* outputStream, int* pixelBaseIdx, int width, int height, Texture2D tex);

	extern __global__ void ROP(const FragmentPSOut* fragmentStream, const int* pixelBaseIdx, int width, int height, unsigned* depthBuffer, float4* renderTarget);

	extern __global__ void StreamingToFrameBuffer(int dimension, float4* renderTarget, unsigned char* framebuffer, int width);
}