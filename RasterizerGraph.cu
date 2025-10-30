#include "GLFW/glfw3.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "Rasterizer.cuh"
#include "RasterizerGraph.h"
#include "RasterUnitFunction.cuh"
#include "RasterParallelAlgorithm.cuh"


namespace
{
	cudaStream_t workStreams[16];
	cudaStream_t renderingStream;
	cudaStream_t glStream;

	cudaGraph_t renderGraph = nullptr;
	cudaGraphExec_t graphInstance = nullptr;

	cudaEvent_t events[16];
}

using namespace CRPipeline;

void InitializeCudaGraph()
{
	for (int i = 0; i < 16; i++)
	{
		CUDA_CHECK(cudaStreamCreateWithFlags(&workStreams[i], cudaStreamNonBlocking));
		CUDA_CHECK(cudaEventCreate(&events[i]));
	}
	CUDA_CHECK(cudaStreamCreateWithFlags(&renderingStream, cudaStreamNonBlocking));
	CUDA_CHECK(cudaStreamCreateWithFlags(&glStream, cudaStreamNonBlocking));
}

void CleanupCudaGraph()
{
	for (int i = 0; i < 16; i++)
	{
		CUDA_CHECK(cudaStreamDestroy(workStreams[i]));
		CUDA_CHECK(cudaEventDestroy(events[i]));
	}
	CUDA_CHECK(cudaStreamDestroy(renderingStream));
	CUDA_CHECK(cudaStreamDestroy(glStream));

	if (renderGraph != nullptr)
	{
		CUDA_CHECK(cudaGraphDestroy(renderGraph));
		renderGraph = nullptr;
	}
	if (graphInstance != nullptr)
	{
		CUDA_CHECK(cudaGraphExecDestroy(graphInstance));
		graphInstance = nullptr;
	}
}

void BeginFrame()
{

	// Reset allocators
	cudaStreamWaitEvent(workStreams[2], events[0], 0); // Wait for SetGraphicsRoot
	CUDA_CHECK(cudaMemsetAsync(dChunkAllocator, 0, sizeof(unsigned int), workStreams[2]));
	CUDA_CHECK(cudaEventRecord(events[2], workStreams[2]));

	cudaStreamWaitEvent(workStreams[3], events[0], 0); // Wait for SetGraphicsRoot
	CUDA_CHECK(cudaMemsetAsync(dTileChunkAllocator, 0, sizeof(unsigned int), workStreams[3]));
	CUDA_CHECK(cudaEventRecord(events[3], workStreams[3]));

	cudaStreamWaitEvent(workStreams[4], events[0], 0); // Wait for SetGraphicsRoot
	CUDA_CHECK(cudaMemsetAsync(dQuadAllocator, 0, sizeof(unsigned int), workStreams[4]));
	CUDA_CHECK(cudaEventRecord(events[4], workStreams[4]));

	cudaStreamWaitEvent(workStreams[5], events[0], 0); // Wait for SetGraphicsRoot
	CUDA_CHECK(cudaMemcpyAsync(dSubTriangleCounter, hdcPrimitiveCount, sizeof(unsigned int), cudaMemcpyHostToDevice, workStreams[5]));
	CUDA_CHECK(cudaEventRecord(events[5], workStreams[5]));

	// Clear depth buffer & render target & hiz
	cudaStreamWaitEvent(workStreams[6], events[0], 0); // Wait for SetGraphicsRoot
	cudaMemsetAsync(dFragmentStream, 0, sizeof(FragmentPSin) * windowWidth * windowHeight * AVERAGE_OVERDRAW, workStreams[6]);
	cudaEventRecord(events[6], workStreams[6]);

	cudaStreamWaitEvent(workStreams[7], events[0], 0); // Wait for SetGraphicsRoot
	cudaMemsetAsync(dRenderTarget, 0, sizeof(float4) * windowHeight * windowWidth, workStreams[7]);
	cudaEventRecord(events[7], workStreams[7]);

	cudaStreamWaitEvent(workStreams[8], events[0], 0); // Wait for SetGraphicsRoot
	cudaMemsetAsync(dHiZ, 0xFF, sizeof(unsigned) * MAX_BIN_COUNT * MAX_TILE_COUNT, workStreams[8]);
	cudaEventRecord(events[8], workStreams[8]);

	cudaStreamWaitEvent(workStreams[9], events[0], 0); // Wait for SetGraphicsRoot
	cudaMemsetAsync(dDepthBuffer, 0xFF, sizeof(unsigned int) * windowWidth * windowHeight, workStreams[9]);
	cudaEventRecord(events[9], workStreams[9]);

	// Clear inner buffers
	cudaStreamWaitEvent(workStreams[10], events[0], 0); // Wait for SetGraphicsRoot
	cudaMemsetAsync(dPrimitiveStream, 0xFF, sizeof(Primitive) * dcPrimitiveCount * 4, workStreams[10]);
	cudaEventRecord(events[10], workStreams[10]);

	cudaStreamWaitEvent(workStreams[12], events[0], 0); // Wait for SetGraphicsRoot
	cudaMemsetAsync(dPixelBaseIdx, 0xFF, sizeof(int) * windowWidth * windowHeight, workStreams[12]);
	cudaEventRecord(events[12], workStreams[12]);

	// Clear queue counters
	cudaStreamWaitEvent(workStreams[13], events[0], 0); // Wait for SetGraphicsRoot
	cudaMemsetAsync(dSubQueuePrimCount, 0, sizeof(unsigned) * MAX_BINNING_WAVE * BINNING_STAGE_BLOCK_SIZE, workStreams[13]);
	cudaEventRecord(events[13], workStreams[13]);

	cudaStreamWaitEvent(workStreams[14], events[0], 0); // Wait for SetGraphicsRoot
	cudaMemsetAsync(dTileQueuePrimCount, 0, sizeof(unsigned) * MAX_BIN_COUNT * MAX_TILE_COUNT * TILE_QUEUE_ENTRY, workStreams[14]);
	cudaEventRecord(events[14], workStreams[14]);
}

void EndFrame()
{
}

void RenderPipeline(Texture2D tex)
{
	const int threadsPerBlock = 256;

	// ===== VERTEX SHADING STAGE =====
	{
		int blocksPerGrid = (dcVertexCount + threadsPerBlock - 1) / threadsPerBlock;
		VertexFetchAndShading << <blocksPerGrid, threadsPerBlock, 0, renderingStream >> > (dcVertexCount, dInVertexStream, dOutVertexStream);
	}

	// ===== PRIMITIVE ASSEMBLY STAGE =====
	{
		CUDA_CHECK(cudaStreamWaitEvent(renderingStream, events[5], 0)); // Wait for SubTriangleCounter memcpy
		cudaStreamWaitEvent(renderingStream, events[10], 0); // Wait for PrimitiveStream memset
		int blocksPerGrid = (dcPrimitiveCount + threadsPerBlock - 1) / threadsPerBlock;
		PrimitiveAssembly << <blocksPerGrid, threadsPerBlock, 0, renderingStream >> > (dcPrimitiveCount, dIndexStream, dOutVertexStream,
			dPrimitiveStream, windowWidth, windowHeight, dSubTriangleCounter);
	}

	// ===== PRIMITIVE COMPACTION STAGE =====
	{
		PrimitiveCompaction(AVERAGE_PRIMITIVE_CULLED_COUNT * dcPrimitiveCount, dPrimitiveStream, dCompactedPrimitiveStream, dPrimitiveCounter, renderingStream);
		cudaEventRecord(events[1], renderingStream);
	}

	// ===== TRIANGLE SETUP STAGE &  BINNING STAGE=====
	{
		unsigned compactedCount = std::max(1u * dcPrimitiveCount * AVERAGE_PRIMITIVE_CULLED_COUNT >> 1, 2048u);
		int blocksPerGrid = (compactedCount + threadsPerBlock - 1) / threadsPerBlock;
		cudaStreamWaitEvent(workStreams[15], events[1], 0); // Wait for PrimitiveCompaction
		TriangleSetup << <blocksPerGrid, threadsPerBlock, 0, workStreams[15] >> > (dPrimitiveCounter, dCompactedPrimitiveStream, dTriSetupData);
		cudaEventRecord(events[15], workStreams[15]);

		cudaStreamWaitEvent(renderingStream, events[13], 0); // Wait for SubQueuePrimCount memset
		cudaStreamWaitEvent(renderingStream, events[2], 0); // Wait for ChunkAllocator memset
		PrimitiveBinning << <blocksPerGrid, threadsPerBlock, 0, renderingStream >> > (dPrimitiveCounter, dCompactedPrimitiveStream, dChunkAllocator,
			dSubQueueBaseIndex, dSubQueuePrimCount, dBinQueue, windowWidth, windowHeight);
	}

	// ===== COARSE RASTERIZATION STAGE =====
	{
		int xUpper = UPPER_BOUND(windowWidth, BIN_PIXEL_SIZE_LOG2);
		int yUpper = UPPER_BOUND(windowHeight, BIN_PIXEL_SIZE_LOG2);
		dim3 blockSize(xUpper >> BIN_PIXEL_SIZE_LOG2, yUpper >> BIN_PIXEL_SIZE_LOG2);
		cudaStreamWaitEvent(renderingStream, events[14], 0); // Wait for TileQueuePrimCount memset
		cudaStreamWaitEvent(renderingStream, events[3], 0); // Wait for TileChunkAllocator memset
		cudaStreamWaitEvent(renderingStream, events[8], 0); // Wait for HiZ memset
		cudaStreamWaitEvent(renderingStream, events[15], 0); // Wait for TriangleSetup
		CoarseRasterizer << <blockSize, dim3(16, 16), 0, renderingStream >> > (dPrimitiveCounter, dSubQueueBaseIndex, dSubQueuePrimCount, dBinQueue,
			dTriSetupData, dHiZ, dTileChunkAllocator, dTileQueueBaseIndex, dTileQueuePrimCount, dTileQueue, windowWidth, windowHeight);
	}

	// ===== FINE RASTERIZATION STAGE =====
	{
		int tileXUpper = UPPER_BOUND(windowWidth, TILE_PIXEL_SIZE_LOG2) >> TILE_PIXEL_SIZE_LOG2;
		int tileYUpper = UPPER_BOUND(windowHeight, TILE_PIXEL_SIZE_LOG2) >> TILE_PIXEL_SIZE_LOG2 >> 2;
		cudaStreamWaitEvent(renderingStream, events[4], 0); // Wait for QuadAllocator memset
		cudaStreamWaitEvent(renderingStream, events[9], 0); // Wait for DepthBuffer memset
		cudaStreamWaitEvent(renderingStream, events[6], 0); // Wait for FragmentStream memset
		FineRasterizerWIP << <dim3(tileXUpper, tileYUpper), dim3(32, FINE_RASTER_TILE_PER_BLOCK), 0, renderingStream >> > (dTileQueueBaseIndex, dTileQueuePrimCount, dTileQueue,
			dTriSetupData, dDepthBuffer, dQuadAllocator, dFragmentStream, windowWidth, windowHeight);
	}

	// ===== PIXEL SHADING STAGE =====
	{
		int blocksPerGrid = (windowWidth * windowHeight * AVERAGE_OVERDRAW + threadsPerBlock - 1) / threadsPerBlock;
		cudaStreamWaitEvent(renderingStream, events[12], 0); // wait for PixelBaseIdx memset
		PixelShader << <blocksPerGrid, threadsPerBlock, 0, renderingStream >> > (dQuadAllocator, dFragmentStream, dFragmentOutStream, dPixelBaseIdx, windowWidth, windowHeight, tex);
	}

	// ===== ROP STAGE =====
	{
		dim3 blockSize(16, 16);
		dim3 gridSize((windowWidth + blockSize.x - 1) / blockSize.x, (windowHeight + blockSize.y - 1) / blockSize.y);
		cudaStreamWaitEvent(renderingStream, events[7], 0); // Wait for RenderTarget memset
		ROP << <gridSize, blockSize, 0, renderingStream >> > (dFragmentOutStream, dPixelBaseIdx, windowWidth, windowHeight, dDepthBuffer, dRenderTarget);
	}


}

void BuildPipeline(unsigned char* cudaMappedRT, unsigned* depthBuffer,
	const VertexVSIn* vertexStream, const uint32_t* indexStream,
	int indexCount, int vertexCount, MatricesCBuffer* cb, Texture2D tex)
{
	CUDA_CHECK(cudaStreamBeginCapture(renderingStream, cudaStreamCaptureModeGlobal));

	cudaEventRecord(events[0], renderingStream);
	SetGraphicsRoot(cudaMappedRT, depthBuffer,
		vertexStream, indexStream,
		indexCount, vertexCount, cb, renderingStream);
	cudaEventRecord(events[1], renderingStream);
	BeginFrame();
	RenderPipeline(tex);
	EndFrame();

	CUDA_CHECK(cudaStreamEndCapture(renderingStream, &renderGraph));
	CUDA_CHECK(cudaGraphInstantiate(&graphInstance, renderGraph, nullptr, nullptr, 0));
}

void RasterizeWithGraph(GLuint rtBuffer, unsigned* depthBuffer,
	const VertexVSIn* vertexStream, const uint32_t* indexStream,
	int indexCount, int vertexCount, MatricesCBuffer* cb, Texture2D tex)
{
	cudaEvent_t renderPipeEvent;
	CUDA_CHECK(cudaEventCreate(&renderPipeEvent));

	CUDA_CHECK(cudaGraphLaunch(graphInstance, renderingStream));
	CUDA_CHECK(cudaEventRecord(renderPipeEvent, renderingStream));

	unsigned char* cudaMappedRT = nullptr;
	CUDA_CHECK(cudaGLMapBufferObjectAsync((void**)&cudaMappedRT, rtBuffer, glStream));

	int blocksPerGrid = (windowWidth * windowHeight + 255) / 256;
	cudaStreamWaitEvent(glStream, renderPipeEvent, 0); // Wait for RT mapping
	StreamingToFrameBuffer << <blocksPerGrid, 256, 0, glStream >> > (windowWidth * windowHeight, dRenderTarget, cudaMappedRT, windowWidth);

	CUDA_CHECK(cudaGLUnmapBufferObjectAsync(rtBuffer, glStream));

	CUDA_CHECK(cudaStreamSynchronize(glStream));
}
