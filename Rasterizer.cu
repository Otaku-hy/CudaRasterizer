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


	////inner variables
	int* dSubTriangleCounter = nullptr;
	Primitive* dCompactedPrimitiveStream = nullptr;
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

__global__ void PrimitiveAssembly(int dimension, const uint32_t* indexStream, const VertexVSOut* vertexStream, Primitive* primitiveStream, int width, int height, int* subTriAllocationCounter)
{
	//TODO : maybe change to guard band clipping instead of view frustum clipping

	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (tidx >= dimension) return;

	uint32_t i0 = indexStream[tidx * 3 + 0];
	uint32_t i1 = indexStream[tidx * 3 + 1];
	uint32_t i2 = indexStream[tidx * 3 + 2];

	VertexVSOut v0 = vertexStream[i0];
	VertexVSOut v1 = vertexStream[i1];
	VertexVSOut v2 = vertexStream[i2];

	int primitiveID = tidx;

	const float epsilon = 1e-6f;
	if (v0.sv_position.w <= epsilon || v1.sv_position.w <= epsilon || v2.sv_position.w <= epsilon)
	{
		primitiveStream[tidx].primitiveID = -1; // Mark as culled
		return;
	}

	// clipping & frustum culling
	VertexVSOut temp1[MAX_VERTEX_CLIP_COUNT];
	VertexVSOut temp2[MAX_VERTEX_CLIP_COUNT];
	temp1[0] = v0;
	temp1[1] = v1;
	temp1[2] = v2;

	int vertexCount = 3;
	vertexCount = ClippingWithPlane({ 1,0,0,1 }, vertexCount, temp1, temp2);
	vertexCount = ClippingWithPlane({ -1,0,0,1 }, vertexCount, temp2, temp1);
	vertexCount = ClippingWithPlane({ 0,1,0,1 }, vertexCount, temp1, temp2);
	vertexCount = ClippingWithPlane({ 0,-1,0,1 }, vertexCount, temp2, temp1);
	vertexCount = ClippingWithPlane({ 0,0,1,1 }, vertexCount, temp1, temp2);
	vertexCount = ClippingWithPlane({ 0,0,-1,1 }, vertexCount, temp2, temp1);

	int numPrimitives = (vertexCount - 2); // triangle fan
	int basePos = tidx;
	if (numPrimitives < 1)
	{
		primitiveID = -1; // culled
		primitiveStream[tidx].primitiveID = primitiveID;
		return;
	}

	for (int i = 0; i < vertexCount; i++)
	{
		temp1[i].sv_position.x /= temp1[i].sv_position.w; 
		temp1[i].sv_position.y /= temp1[i].sv_position.w;
		temp1[i].sv_position.z /= temp1[i].sv_position.w; // perspective divide
		temp1[i].sv_position.x = (temp1[i].sv_position.x + 1.0f) * 0.5f * width;
		temp1[i].sv_position.y = (temp1[i].sv_position.y + 1.0f) * 0.5f * height;
	}

	////maybe optimize here: a block use one global atomicAdd, the threads use shared mem atomicAdd
	if (numPrimitives > 1) 
	{
		primitiveStream[basePos].primitiveID = -1;
		basePos = atomicAdd(subTriAllocationCounter, numPrimitives);
	}

	for (int j = 1; j < vertexCount-1; j++)
	{
		glm::vec2 e0 = glm::vec2(temp1[j].sv_position) - glm::vec2(temp1[0].sv_position);
		glm::vec2 e1 = glm::vec2(temp1[j + 1].sv_position) - glm::vec2(temp1[0].sv_position);
		float area = 0.5f * (e0.x * e1.y - e0.y * e1.x);
		primitiveID = area > 0 ? tidx : -1; // backface culled 
		primitiveStream[basePos + j - 1] = { {temp1[0], temp1[j], temp1[j + 1]}, primitiveID };
	}
}

__global__ void Rasterization(int dimension, const Primitive* primitiveStream, FragmentPSin* fragmentStream, float* depthBuffer, int width, int height)
{
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (tidx >= dimension) return;

	Primitive prim = primitiveStream[tidx];
	if (prim.primitiveID == -1) return;

	glm::vec2 v0 = glm::vec2(prim.v[0].sv_position);
	glm::vec2 v1 = glm::vec2(prim.v[1].sv_position);
	glm::vec2 v2 = glm::vec2(prim.v[2].sv_position);

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

			float zInterpolated = 1.0f / (barycentric.x / d0 + barycentric.y / d1 + barycentric.z / d2);
			int pixelIndex = y * width + x;

			float depthOld = depthBuffer[pixelIndex];
			if (zInterpolated >= depthOld) continue;

			depthOld = atomicMinFloat(&depthBuffer[pixelIndex], zInterpolated);
			if (depthOld < zInterpolated) continue;

			// interpolate attributes, just a example here
			FragmentPSin fragment;
			fragment.color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
			fragmentStream[pixelIndex] = fragment;
		}
	}
}

__global__ void PrimitiveBinning(int size, Primitive* primitiveStream)
{
	/// ? why need this
	__shared__ unsigned int binIntesectMask[BINNING_BLOCK_SIZE][BIN_MAX_COUNT >> 5 + 1];


	__shared__ unsigned int binTriangleCount[BIN_MAX_COUNT];

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= size) return;

	Primitive prim = primitiveStream[idx];
	AABB<glm::vec2> boundingBox = ComputeTriangleBoundingBox(glm::vec2(prim.v[0].sv_position), glm::vec2(prim.v[1].sv_position), glm::vec2(prim.v[2].sv_position));
	
	unsigned int intersectMask[BIN_MAX_COUNT >> 5];
	for (int i = 0; i < (BIN_MAX_COUNT >> 5); i++) intersectMask[i] = 0;

	unsigned int lox = 
	

	//
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

	CUDA_CHECK(cudaMalloc((void**)&dSubTriangleCounter, sizeof(int)));

	InitCompactionEnvironment();
}

void RasterizerUpdateObjectsBuffer(int indexCountPerPrimitive, int vertexCount, int indexCount)
{
	if (dOutVertexStream) CUDA_CHECK(cudaFree(dOutVertexStream));
	if (dPrimitiveStream) CUDA_CHECK(cudaFree(dPrimitiveStream));
	if (dCompactedPrimitiveStream) CUDA_CHECK(cudaFree(dCompactedPrimitiveStream));

	//only support triangle as primitive now
	// so premitiveCount = indexCount / 3
	int primitiveCount = (indexCount + indexCountPerPrimitive - 1) / indexCountPerPrimitive;

	CUDA_CHECK(cudaMalloc((void**)&dOutVertexStream, sizeof(VertexVSOut) * vertexCount));
	CUDA_CHECK(cudaMalloc((void**)&dPrimitiveStream, sizeof(Primitive) * primitiveCount * 4)); // allocate 2 time for clipping case
	CUDA_CHECK(cudaMalloc((void**)&dCompactedPrimitiveStream, sizeof(Primitive) * primitiveCount * 4));
}

void CleanupCudaRasterizer()
{
	DestroyCompactionEnvironment();

	cudaFree(dFragmentStream);
	cudaFree(dRenderTarget);
	cudaFree(dOutVertexStream);
	cudaFree(dPrimitiveStream);
	cudaFree(dSubTriangleCounter);
	cudaFree(dCompactedPrimitiveStream);
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

	CUDA_CHECK(cudaMemcpy(dSubTriangleCounter, &dcPrimitiveCount, sizeof(int), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpyToSymbol(cbVertex, cb, sizeof(MatricesCBuffer)));

	int threadsPerBlock = 256;

	// clear depth buffer & render target
	CUDA_CHECK(cudaMemset(dFragmentStream, 0, sizeof(FragmentPSin) * windowWidth * windowHeight));
	ClearDepthBuffer << <(windowWidth * windowHeight + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (windowWidth * windowHeight, dDepthBuffer);

	//clear inner buffers
	CUDA_CHECK(cudaMemset(dPrimitiveStream, 0xFF, sizeof(Primitive) * dcPrimitiveCount * 4));
	CUDA_CHECK(cudaMemset(dOutVertexStream, 0, sizeof(VertexVSOut) * dcVertexCount));

	// vertex fetch and shading
	{
		int blocksPerGrid = (dcVertexCount + threadsPerBlock - 1) / threadsPerBlock;
		VertexFetchAndShading << <blocksPerGrid, threadsPerBlock >> > (dcVertexCount, dInVertexStream, dOutVertexStream);
	}

	// primitive assembly
	int primitiveCount = 0;
	{
		int blocksPerGrid = (dcPrimitiveCount + threadsPerBlock - 1) / threadsPerBlock;
		PrimitiveAssembly << <blocksPerGrid, threadsPerBlock >> > (dcPrimitiveCount, dIndexStream, dOutVertexStream, dPrimitiveStream, windowWidth, windowHeight, dSubTriangleCounter);
		primitiveCount = PrimitiveCompaction(4 * dcPrimitiveCount, dPrimitiveStream, dCompactedPrimitiveStream);
		CUDA_CHECK( cudaGetLastError());
	}

	// rasterization
	if (primitiveCount != 0)
	{
		int blocksPerGrid = (primitiveCount + threadsPerBlock - 1) / threadsPerBlock;
		Rasterization << <blocksPerGrid, threadsPerBlock >> > (primitiveCount, dCompactedPrimitiveStream, dFragmentStream, dDepthBuffer, windowWidth, windowHeight);
		CUDA_CHECK(cudaGetLastError());
	}

	// pixel shader
	{
		dim3 blockSize(16, 16);
		dim3 gridSize(
			(windowWidth + blockSize.x - 1) / blockSize.x,
			(windowHeight + blockSize.y - 1) / blockSize.y);
		PixelShader << <gridSize, blockSize >> > (windowWidth * windowHeight, dFragmentStream, dRenderTarget, windowWidth, windowHeight);
		CUDA_CHECK(cudaGetLastError());
	}

	// output merger
	{
		int blocksPerGrid = (windowWidth * windowHeight + threadsPerBlock - 1) / threadsPerBlock;
		OutputMerger << <blocksPerGrid, threadsPerBlock >> > (windowWidth * windowHeight, dRenderTarget, dFrameBuffer, windowWidth);
	}


}




