#include "Rasterizer.h"

#include "RasterUnitFunction.cuh"

/*buffer properties :
	inVertexStream: set through host
	outVertexStream: rasterizer innner buffer
	indexStream: set through host
	primitiveStream: rasterizer innner buffer
	fragmentStream: rasterizer innner buffer
	depthBuffer: set through host
	renderTarget: rasterizer innner buffer
	framebuffer: set through host

	set while intialize: renderTarget  & fragmentStream (boath screen size)
	set while obj change: outVertexStream (obj vertex count), primitiveStream (obj triangle count)
	set through draw call: inVertexStream (obj vertex count), indexStream(...), depthBuffer(screen size), framebuffer(screen size)
*/


namespace  //// pipeline signature
{
	__constant__ MatricesCBuffer cbVertex;

	const VertexVSIn* dInVertexStream = nullptr;		// input vertex buffer
	VertexVSOut* dOutVertexStream = nullptr;	// output vertex buffer after vertex shader
	const uint32_t* dIndexStream = nullptr;				// index buffer
	Primitive* dPrimitiveStream = nullptr;		// primitive buffer after primitive assembly
	FragmentPSin* dFragmentStream = nullptr;	// fragment buffer after rasterization
	float* dDepthBuffer = nullptr;			// depth buffer
	glm::vec4* dRenderTarget = nullptr;		// render target buffer after pixel shader
	unsigned char* dFrameBuffer;		// output framebuffer

	int windowWidth = 0;
	int windowHeight = 0;

	int dcVertexCount = 0;		// current obj vertex count
	int dcIndexCount = 0;			// current obj index count
	int dcIndexCountPerPrimitive = 3; // triangle as primitive now
	int dcPrimitiveCount = 0;		// current obj primitive count
}

__global__ void ClearDepthBuffer(int dimension, float* depthBuffer)
{
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (tidx >= dimension) return;
	depthBuffer[tidx] = 1.0f;
}

__global__ void PrimitiveDistribution();

__global__ void VertexFetchAndShading(int dimension, const VertexVSIn* inVertexStream, VertexVSOut* outVertexStream)
{
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (tidx >= dimension) return;

	// vertex fetch
	VertexVSIn input = inVertexStream[tidx];

	VertexVSOut output;
	output.sv_position = cbVertex.mvp * input.position;

	outVertexStream[tidx] = output;
}

__global__ void PrimitiveAssembly(int dimension, const VertexVSOut* vertexStream, const uint32_t* indexStream, Primitive* primitiveStream)
{
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (tidx >= dimension) return;

	int primIndex = tidx / 3;
	int indexInPrim = tidx - primIndex * 3;
	primitiveStream[primIndex].v[indexInPrim] = vertexStream[indexStream[tidx]];
}

__global__ void CullingAndViewportTranform(int dimension, Primitive* primitiveStream, int width, int height)
{
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (tidx >= dimension) return;

	Primitive prim = primitiveStream[tidx];

	glm::vec4 v0 = prim.v[0].sv_position;
	glm::vec4 v1 = prim.v[1].sv_position;
	glm::vec4 v2 = prim.v[2].sv_position;

	//// perspective divide
	v0 /= v0.w;
	v1 /= v1.w;
	v2 /= v2.w;

	//// ndc & viewport transform
	v0.x = (v0.x + 1.0f) * 0.5f * width;
	v0.y = (v0.y + 1.0f) * 0.5f * height;
	v1.x = (v1.x + 1.0f) * 0.5f * width;
	v1.y = (v1.y + 1.0f) * 0.5f * height;
	v2.x = (v2.x + 1.0f) * 0.5f * width;
	v2.y = (v2.y + 1.0f) * 0.5f * height;

	prim.v[0].sv_position = v0;
	prim.v[1].sv_position = v1;
	prim.v[2].sv_position = v2;

	primitiveStream[tidx] = prim;
}

__global__ void Rasterization(int dimension, Primitive* primitiveStream, FragmentPSin* fragmentStream, float* depthBuffer, int width, int height)
{
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (tidx >= dimension) return;

	Primitive prim = primitiveStream[tidx];

	glm::vec2 v0 = { prim.v[0].sv_position.x, prim.v[0].sv_position.y};
	glm::vec2 v1 = { prim.v[1].sv_position.x, prim.v[1].sv_position.y };
	glm::vec2 v2 = { prim.v[2].sv_position.x, prim.v[2].sv_position.y };

	float d0 = prim.v[0].sv_position.z;
	float d1 = prim.v[1].sv_position.z;
	float d2 = prim.v[2].sv_position.z;

	AABB<glm::vec2> bounding = ComputeTriangleBoundingBox(v0, v1, v2);
	int minX = glm::clamp((int)bounding.min.x, 0, width - 1);
	int minY = glm::clamp((int)bounding.min.y, 0, height - 1);
	int maxX = glm::clamp((int)bounding.max.x, 0, width - 1);
	int maxY = glm::clamp((int)bounding.max.y, 0, height - 1);

	for (int y = minY; y <= maxY; y++)
	{
		for (int x = minX; x <= maxX; x++)
		{
			glm::vec2 pixelCenter = glm::vec2(x + 0.5f, y + 0.5f);
			glm::vec3 barycentric = ComputeBarycentric2D(pixelCenter, v0, v1, v2);
			if (barycentric.x < 0 || barycentric.y < 0 || barycentric.z < 0) continue;

			float zInterpolated = 1.0f / (barycentric.x/d0 + barycentric.y/d1 + barycentric.z/d2);
			int pixelIndex = y * width + x;

			float depthOld = depthBuffer[pixelIndex];
			if (zInterpolated >= depthOld) continue;

			depthOld = atomicMinFloat(&depthBuffer[pixelIndex], zInterpolated);
			if (depthOld < zInterpolated) continue;

			// interpolate attributes, just a example here
			FragmentPSin fragment;
			fragment.color = glm::vec4(1.0f, 1.0f, 1.0f,1.0f);
			fragmentStream[pixelIndex] = fragment;
		}
	}
}

__global__ void PixelShader(int dimension, FragmentPSin* fragmentStream, glm::vec4* renderTarget, int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;

	int pixelIndex = y * width + x;

	FragmentPSin fragment = fragmentStream[pixelIndex];
	renderTarget[pixelIndex] = fragment.color;
}

//__global__ void OutputMerger(int dimension, glm::vec4* renderTarget, cudaSurfaceObject_t outputSurface, int width)
//{
//	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
//	if (tidx >= dimension) return;
//
//	glm::vec4 color = renderTarget[tidx];
//
//	int y = tidx / width;
//	int x = tidx - y * width;
//	
//	uchar4 outputColor = make_uchar4(
//		(unsigned char)(glm::clamp(color.r, 0.0f, 1.0f) * 255),
//		(unsigned char)(glm::clamp(color.g, 0.0f, 1.0f) * 255),
//		(unsigned char)(glm::clamp(color.b, 0.0f, 1.0f) * 255),
//		(unsigned char)(glm::clamp(color.a, 0.0f, 1.0f) * 255)
//	);
//
//	surf2Dwrite(outputColor, outputSurface, x * sizeof(uchar4), y);
//}

__global__ void OutputMerger(int dimension, glm::vec4* renderTarget, unsigned char* framebuffer, int width)
{
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (tidx >= dimension) return;

	glm::vec4 color = renderTarget[tidx];

	int y = tidx / width;
	int x = tidx - y * width;
	
	unsigned char r = (unsigned char)(glm::clamp(color.r, 0.0f, 1.0f) * 255.0f);
	unsigned char g = (unsigned char)(glm::clamp(color.g, 0.0f, 1.0f) * 255.0f);
	unsigned char b = (unsigned char)(glm::clamp(color.b, 0.0f, 1.0f) * 255.0f);
	unsigned char a = (unsigned char)(glm::clamp(color.a, 0.0f, 1.0f) * 255.0f);

	framebuffer[tidx * 4 + 0] = r; // Red
	framebuffer[tidx * 4 + 1] = g; // Green
	framebuffer[tidx * 4 + 2] = b; // Blue
	framebuffer[tidx * 4 + 3] = a; // Alpha
}


void InitializeCudaRasterizer(int width, int height)
{
	windowHeight = height;
	windowWidth = width;

	CUDA_CHECK(cudaMalloc((void**)&dFragmentStream, sizeof(FragmentPSin) * windowHeight * windowWidth));
	CUDA_CHECK(cudaMalloc((void**)&dRenderTarget, sizeof(glm::vec4) * windowHeight * windowWidth));
}

void RasterizerUpdateObjectsBuffer(int indexCountPerPrimitive, int vertexCount, int indexCount)
{
	if (dOutVertexStream) CUDA_CHECK(cudaFree(dOutVertexStream));
	if (dPrimitiveStream) CUDA_CHECK(cudaFree(dPrimitiveStream));

	//only support triangle as primitive now
	// so premitiveCount = indexCount / 3
	int primitiveCount = (indexCount + indexCountPerPrimitive -1)/ indexCountPerPrimitive;

	CUDA_CHECK(cudaMalloc((void**)&dOutVertexStream,sizeof(VertexVSOut) * vertexCount));
	CUDA_CHECK(cudaMalloc((void**)&dPrimitiveStream, sizeof(Primitive) * primitiveCount));
}

void CleanupCudaRasterizer()
{
	cudaFree(dFragmentStream);
	cudaFree(dRenderTarget);
	cudaFree(dOutVertexStream);
	cudaFree(dPrimitiveStream);
}

void Rasterize(unsigned char* outRenderTarget, float* depthBuffer,
	const VertexVSIn* vertexStream, const uint32_t* indexStream, 
	int indexCount, int vertexCount, MatricesCBuffer* cb)
{
	dFrameBuffer = outRenderTarget;
	dDepthBuffer = depthBuffer;
	dInVertexStream = vertexStream;
	dIndexStream = indexStream;

	dcIndexCount = indexCount;
	dcVertexCount = vertexCount;
	dcPrimitiveCount = (indexCount + dcIndexCountPerPrimitive - 1) / dcIndexCountPerPrimitive;

	CUDA_CHECK(cudaMemcpyToSymbol(cbVertex, cb, sizeof(MatricesCBuffer)));

	int threadsPerBlock = 256;

	// clear depth buffer & render target
	cudaMemset(dFragmentStream, 0, sizeof(FragmentPSin) * windowWidth * windowHeight);
	ClearDepthBuffer << <(windowWidth * windowHeight + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (windowWidth * windowHeight, dDepthBuffer);

	// vertex fetch and shading
	{
		int blocksPerGrid = (dcVertexCount + threadsPerBlock - 1) / threadsPerBlock;
		VertexFetchAndShading << <blocksPerGrid, threadsPerBlock >> > (dcVertexCount, dInVertexStream, dOutVertexStream);
	}

	// primitive assembly
	{
		int blocksPerGrid = (dcIndexCount + threadsPerBlock - 1) / threadsPerBlock;
		PrimitiveAssembly << <blocksPerGrid, threadsPerBlock >> > (dcIndexCount, dOutVertexStream, dIndexStream, dPrimitiveStream);
	}

	// culling and viewport transform
	{
		int blocksPerGrid = (dcPrimitiveCount + threadsPerBlock - 1) / threadsPerBlock;
		CullingAndViewportTranform << <blocksPerGrid, threadsPerBlock >> > (dcPrimitiveCount, dPrimitiveStream, windowWidth, windowHeight);
	}

	// rasterization
	{
		int blocksPerGrid = (dcPrimitiveCount + threadsPerBlock - 1) / threadsPerBlock;
		Rasterization << <blocksPerGrid, threadsPerBlock >> > (dcPrimitiveCount, dPrimitiveStream, dFragmentStream, dDepthBuffer, windowWidth, windowHeight);
	}

	// pixel shader
	{
		dim3 blockSize(16, 16);
		dim3 gridSize(
			(windowWidth + blockSize.x - 1) / blockSize.x,
			(windowHeight + blockSize.y - 1) / blockSize.y);
		PixelShader << <gridSize, blockSize >> > (windowWidth * windowHeight, dFragmentStream, dRenderTarget, windowWidth, windowHeight);
	}

	// output merger
	{
		int blocksPerGrid = (windowWidth * windowHeight + threadsPerBlock - 1) / threadsPerBlock;
		OutputMerger << <blocksPerGrid, threadsPerBlock >> > (windowWidth * windowHeight, dRenderTarget, dFrameBuffer, windowWidth);
	}
}




