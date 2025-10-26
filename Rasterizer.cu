#include "Rasterizer.h"
#include "Rasterizer.cuh"

/*buffer properties :
	inVertexStream: set through host
	outVertexStream: rasterizer innner buffer
	indexStream: set through host
	primitiveStream: rasterizer innner buffer
	fragmentStream: rasterizer innner buffer
	depthBuffer: set through host
	renderTarget: rasterizer innner buffer
	framebuffer: set through host

	set while intialize: renderTarget  (screen size)
	set while obj change: outVertexStream (obj vertex count), primitiveStream (obj triangle count)
	set through draw call: inVertexStream (obj vertex count), indexStream(...), depthBuffer(screen size), framebuffer(screen size)
*/

__constant__ MatricesCBuffer cbVertex;

namespace CRPipeline//// pipeline signature
{
	const VertexVSIn* dInVertexStream = nullptr;		// input vertex buffer
	VertexVSOut* dOutVertexStream = nullptr;	// output vertex buffer after vertex shader
	const uint32_t* dIndexStream = nullptr;				// index buffer
	Primitive* dPrimitiveStream = nullptr;		// primitive buffer after primitive assembly
	FragmentPSin* dFragmentStream = nullptr;	// fragment buffer after rasterization
	FragmentPSOut* dFragmentOutStream = nullptr; // fragment info used for ROP
	unsigned* dDepthBuffer = nullptr;
	float4* dRenderTarget = nullptr;		// render target buffer after pixel shader
	unsigned char* dFrameBuffer;		// output framebuffer
	unsigned int* dHiZ = nullptr;

	int windowWidth = 0;
	int windowHeight = 0;

	int dcVertexCount = 0;		// current obj vertex count
	int dcIndexCount = 0;			// current obj index count
	int dcIndexCountPerPrimitive = 3; // triangle as primitive now
	int dcPrimitiveCount = 0;		// current obj primitive count

	////inner variables
	TriangleSetupData* dTriSetupData = nullptr;
	Primitive* dCompactedPrimitiveStream = nullptr;
	int* dPixelBaseIdx = nullptr;

	unsigned int* dSubTriangleCounter = nullptr; //primitive count after culling before compaction
	unsigned int* dTrunkAllocator = nullptr;
	unsigned int* dTileTrunkAllocator = nullptr;
	unsigned int* dQuadAllocator = nullptr;
	unsigned int* dBinSubQueueCounter = nullptr;
	unsigned int* dPrimitiveCounter = nullptr; //primitive count after compaction
	//pined host memory
	unsigned int* hTileTrunkCount = nullptr;
	unsigned int* hBinTrunkCount = nullptr;
	unsigned int* hFragmentCount = nullptr;
	unsigned int* hdcPrimitiveCount = nullptr;

	unsigned int* dSubQueueBaseIndex = nullptr;
	unsigned int* dSubQueuePrimCount = nullptr;
	unsigned int* dBinQueue = nullptr;
	unsigned int* dTileQueueBaseIndex = nullptr;
	unsigned int* dTileQueuePrimCount = nullptr;
	unsigned int* dTileQueue = nullptr;

}

namespace CRPipeline
{
	void SetGraphicsRoot(unsigned char* outRenderTarget, unsigned* depthBuffer,
		const VertexVSIn* vertexStream, const uint32_t* indexStream,
		int indexCount, int vertexCount, MatricesCBuffer* cb, cudaStream_t stream)
	{
		dFrameBuffer = outRenderTarget;
		dDepthBuffer = depthBuffer;
		dInVertexStream = vertexStream;
		dIndexStream = indexStream;
		CUDA_CHECK(cudaMemcpyToSymbolAsync(cbVertex, cb, sizeof(MatricesCBuffer),0,cudaMemcpyHostToDevice, stream));

		dcIndexCount = indexCount;
		dcVertexCount = vertexCount;
		dcPrimitiveCount = (indexCount + dcIndexCountPerPrimitive - 1) / dcIndexCountPerPrimitive;
		*hdcPrimitiveCount = dcPrimitiveCount;
		
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
		output.normal = glm::normalize(input.normal); // normalOS = normalWS
		output.uv = input.uv;

		outVertexStream[tidx] = output;
	}

	__global__ void PrimitiveAssembly(int dimension, const uint32_t* indexStream, const VertexVSOut* vertexStream, Primitive* primitiveStream, int width, int height, unsigned int* subTriAllocationCounter)
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

		for (int j = 1; j < vertexCount - 1; j++)
		{
			glm::vec2 e0 = glm::vec2(temp1[j].sv_position) - glm::vec2(temp1[0].sv_position);
			glm::vec2 e1 = glm::vec2(temp1[j + 1].sv_position) - glm::vec2(temp1[0].sv_position);
			float area = 0.5f * (e0.x * e1.y - e0.y * e1.x);
			primitiveID = area > 0 ? tidx : -1; // backface culled 
			primitiveStream[basePos + j - 1] = { {temp1[0], temp1[j], temp1[j + 1]}, primitiveID };
		}
	}

	//__global__ void Rasterization(int dimension, const Primitive* primitiveStream, FragmentPSin* fragmentStream, float* depthBuffer, int width, int height)
	//{
	//	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	//	if (tidx >= dimension) return;
	//
	//	Primitive prim = primitiveStream[tidx];
	//	if (prim.primitiveID == -1) return;
	//
	//	glm::vec2 v0 = glm::vec2(prim.v[0].sv_position);
	//	glm::vec2 v1 = glm::vec2(prim.v[1].sv_position);
	//	glm::vec2 v2 = glm::vec2(prim.v[2].sv_position);
	//
	//	float d0 = prim.v[0].sv_position.z;
	//	float d1 = prim.v[1].sv_position.z;
	//	float d2 = prim.v[2].sv_position.z;
	//
	//	AABB<glm::vec2> bounding = ComputeTriangleBoundingBox(v0, v1, v2);
	//	int minX = glm::clamp((int)bounding.min.x, 0, width - 1);
	//	int minY = glm::clamp((int)bounding.min.y, 0, height - 1);
	//	int maxX = glm::clamp((int)bounding.max.x, 0, width - 1);
	//	int maxY = glm::clamp((int)bounding.max.y, 0, height - 1);
	//
	//	for (int y = minY; y <= maxY; y++)
	//	{
	//		for (int x = minX; x <= maxX; x++)
	//		{
	//			glm::vec2 pixelCenter = glm::vec2(x + 0.5f, y + 0.5f);
	//			glm::vec3 barycentric = ComputeBarycentric2D(pixelCenter, v0, v1, v2);
	//			if (barycentric.x < 0 || barycentric.y < 0 || barycentric.z < 0) continue;
	//
	//			float zInterpolated = 1.0f / (barycentric.x / d0 + barycentric.y / d1 + barycentric.z / d2);
	//			int pixelIndex = y * width + x;
	//
	//			float depthOld = depthBuffer[pixelIndex];
	//			if (zInterpolated >= depthOld) continue;
	//
	//			depthOld = atomicMinFloat(&depthBuffer[pixelIndex], zInterpolated);
	//			if (depthOld < zInterpolated) continue;
	//
	//			// interpolate attributes, just a example here
	//			FragmentPSin fragment;
	//			fragment.color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
	//			fragmentStream[pixelIndex] = fragment;
	//		}
	//	}
	//}

	__global__ void PrimitiveBinning(const unsigned int* pSize, const Primitive* primitiveStream, unsigned int* trunkAllocator, unsigned int* subQueueBaseIndex, unsigned int* subQueuePrimCount, unsigned int* queue, int width, int height)
	{
		__shared__ unsigned sBinTriangleCountWarp[BINNING_STAGE_BLOCK_SIZE / 32 /*warp count*/][MAX_BIN_COUNT + 1];
		__shared__ unsigned sBinMask[BINNING_STAGE_BLOCK_SIZE / 32][MAX_BIN_COUNT + 1];
		__shared__ unsigned sBinTrunkCount[32];
		__shared__ unsigned int sTrunkAllocationBase;

		unsigned int size = *pSize;
		if (size == 0) return;

		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		unsigned int intersectMask[MAX_BIN_COUNT >> 5];
		for (int i = 0; i < (MAX_BIN_COUNT >> 5); i++) intersectMask[i] = 0;

		unsigned int lox, loy, hix, hiy;
		if (idx < size)
		{
			Primitive prim = primitiveStream[idx];
			AABB<glm::vec2> boundingBox = ComputeTriangleBoundingBox(glm::vec2(prim.v[0].sv_position), glm::vec2(prim.v[1].sv_position), glm::vec2(prim.v[2].sv_position));

			unsigned int maxBinX = (width + BIN_PIXEL_SIZE - 1) / BIN_PIXEL_SIZE;
			unsigned int maxBinY = (height + BIN_PIXEL_SIZE - 1) / BIN_PIXEL_SIZE;

			lox = glm::clamp((unsigned int)boundingBox.min.x / BIN_PIXEL_SIZE, 0u, maxBinX);
			loy = glm::clamp((unsigned int)boundingBox.min.y / BIN_PIXEL_SIZE, 0u, maxBinY);
			hix = glm::clamp((unsigned int)boundingBox.max.x / BIN_PIXEL_SIZE, 0u, maxBinX);
			hiy = glm::clamp((unsigned int)boundingBox.max.y / BIN_PIXEL_SIZE, 0u, maxBinY);

			for (int y = loy; y <= hiy; y++)
			{
				for (int x = lox; x <= hix; x++)
				{
					unsigned int binIdx = x + y * 16;
					unsigned int mask = binIdx & 0x1F;
					intersectMask[binIdx >> 5] |= (1 << mask);
				}
			}
		}

		unsigned int warpMask = 0;
		for (int i = 0; i < MAX_BIN_COUNT; i++)
		{
			int laneCounter = i & 0x0000001F;
			bool predicate = (intersectMask[i >> 5] & (1u << laneCounter)) != 0;
			unsigned warpBallot = __ballot_sync(0xFFFFFFFF, predicate);
			if (laneCounter == get_lane_id()) warpMask = warpBallot;
			if (laneCounter == 31)
				sBinMask[threadIdx.x >> 5][i - 31 + get_lane_id()] = warpMask; //first use it as a mask storage
		}
		__syncthreads();

		for (int i = 0; i < MAX_BIN_COUNT >> 5; i++)
		{
			intersectMask[i] = sBinMask[i][threadIdx.x];
		}

		unsigned int prefix = 0;
		for (int i = 0; i < MAX_BIN_COUNT >> 5; i++)
		{
			unsigned int mask = intersectMask[i];
			unsigned int warpSum = __popc(mask);
			prefix += warpSum;
			sBinTriangleCountWarp[i][threadIdx.x] = prefix;
		}
		__syncthreads();

		//output for each bin queue
		unsigned int subQueueSum = sBinTriangleCountWarp[(BINNING_STAGE_BLOCK_SIZE >> 5) - 1][threadIdx.x];
		unsigned int trunkCount = (subQueueSum + QUEUE_TRUNK_SIZE_UINT - 1) / QUEUE_TRUNK_SIZE_UINT;
		__syncthreads();

		if (threadIdx.x < 32) sBinTrunkCount[threadIdx.x] = 0; // clear
		__syncthreads();
		unsigned int prefixTrunkCount = ScanInBlockExclusive(trunkCount, sBinTrunkCount);

		if (threadIdx.x == 255)
		{
			unsigned int totalTrunkCount = prefixTrunkCount + trunkCount;
			sTrunkAllocationBase = atomicAdd(trunkAllocator, totalTrunkCount);
		}
		__syncthreads();

		unsigned int subQueueBase = (prefixTrunkCount + sTrunkAllocationBase) * QUEUE_TRUNK_SIZE_UINT;
		subQueueBaseIndex[idx] = subQueueBase;
		subQueuePrimCount[idx] = subQueueSum;

		// write triangle index to queue
		if (idx >= size) return;

		for (int y = loy; y <= hiy; y++)
		{
			for (int x = lox; x <= hix; x++)
			{
				unsigned int binIdx = x + y * 16;
				unsigned int baseIdx = subQueueBaseIndex[blockIdx.x * blockDim.x + binIdx];
				unsigned int mask = sBinMask[threadIdx.x >> 5][binIdx];
				unsigned int warpSum = __popc(mask);
				unsigned int warpPrefix = sBinTriangleCountWarp[threadIdx.x >> 5][binIdx] - warpSum;
				unsigned int idxInWarp = __popc(mask & ((1u << get_lane_id()) - 1));
				queue[baseIdx + warpPrefix + idxInWarp] = idx; // write triangle index
			}
		}

	}

	__global__ void TriangleSetup(const unsigned int* pSize, const Primitive* primitiveStream, TriangleSetupData* triPreparedStream)
	{
		int size = *pSize;
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx >= size) return;

		Primitive prim = primitiveStream[idx];
		prim.v[0].normal /= prim.v[0].sv_position.w;
		prim.v[1].normal /= prim.v[1].sv_position.w;
		prim.v[2].normal /= prim.v[2].sv_position.w;
		prim.v[0].uv /= prim.v[0].sv_position.w;
		prim.v[1].uv /= prim.v[1].sv_position.w;
		prim.v[2].uv /= prim.v[2].sv_position.w;

		AABB<glm::vec2> bbox = ComputeTriangleBoundingBox(glm::vec2(prim.v[0].sv_position), glm::vec2(prim.v[1].sv_position), glm::vec2(prim.v[2].sv_position));

		TriangleSetupData data;
		data.bounding = glm::vec4(bbox.min, bbox.max);

		float e0a = prim.v[1].sv_position.y - prim.v[2].sv_position.y;
		float e0b = prim.v[2].sv_position.x - prim.v[1].sv_position.x;
		float e0c = prim.v[1].sv_position.x * prim.v[2].sv_position.y - prim.v[2].sv_position.x * prim.v[1].sv_position.y;

		float e1a = prim.v[2].sv_position.y - prim.v[0].sv_position.y;
		float e1b = prim.v[0].sv_position.x - prim.v[2].sv_position.x;
		float e1c = prim.v[2].sv_position.x * prim.v[0].sv_position.y - prim.v[0].sv_position.x * prim.v[2].sv_position.y;

		float e2a = prim.v[0].sv_position.y - prim.v[1].sv_position.y;
		float e2b = prim.v[1].sv_position.x - prim.v[0].sv_position.x;
		float e2c = prim.v[0].sv_position.x * prim.v[1].sv_position.y - prim.v[1].sv_position.x * prim.v[0].sv_position.y;

		float C = e0c + e1c + e2c;

		float zA = (e0a * prim.v[0].sv_position.z + e1a * prim.v[1].sv_position.z + e2a * prim.v[2].sv_position.z) / C;
		float zB = (e0b * prim.v[0].sv_position.z + e1b * prim.v[1].sv_position.z + e2b * prim.v[2].sv_position.z) / C;
		float zD = (e0c * prim.v[0].sv_position.z + e1c * prim.v[1].sv_position.z + e2c * prim.v[2].sv_position.z) / C;

		float oneOverW0 = 1.0f / prim.v[0].sv_position.w;
		float oneOverW1 = 1.0f / prim.v[1].sv_position.w;
		float oneOverW2 = 1.0f / prim.v[2].sv_position.w;
		float oneOverW_A = (e0a * oneOverW0 + e1a * oneOverW1 + e2a * oneOverW2) / C;
		float oneOverW_B = (e0b * oneOverW0 + e1b * oneOverW1 + e2b * oneOverW2) / C;
		float oneOverW_D = (e0c * oneOverW0 + e1c * oneOverW1 + e2c * oneOverW2) / C;

		float normNxA = (e0a * prim.v[0].normal.x + e1a * prim.v[1].normal.x + e2a * prim.v[2].normal.x) / C;
		float normNxB = (e0b * prim.v[0].normal.x + e1b * prim.v[1].normal.x + e2b * prim.v[2].normal.x) / C;
		float normNxD = (e0c * prim.v[0].normal.x + e1c * prim.v[1].normal.x + e2c * prim.v[2].normal.x) / C;

		float normNyA = (e0a * prim.v[0].normal.y + e1a * prim.v[1].normal.y + e2a * prim.v[2].normal.y) / C;
		float normNyB = (e0b * prim.v[0].normal.y + e1b * prim.v[1].normal.y + e2b * prim.v[2].normal.y) / C;
		float normNyD = (e0c * prim.v[0].normal.y + e1c * prim.v[1].normal.y + e2c * prim.v[2].normal.y) / C;

		float normNzA = (e0a * prim.v[0].normal.z + e1a * prim.v[1].normal.z + e2a * prim.v[2].normal.z) / C;
		float normNzB = (e0b * prim.v[0].normal.z + e1b * prim.v[1].normal.z + e2b * prim.v[2].normal.z) / C;
		float normNzD = (e0c * prim.v[0].normal.z + e1c * prim.v[1].normal.z + e2c * prim.v[2].normal.z) / C;

		float uvxA = (e0a * prim.v[0].uv.x + e1a * prim.v[1].uv.x + e2a * prim.v[2].uv.x) / C;
		float uvxB = (e0b * prim.v[0].uv.x + e1b * prim.v[1].uv.x + e2b * prim.v[2].uv.x) / C;
		float uvxD = (e0c * prim.v[0].uv.x + e1c * prim.v[1].uv.x + e2c * prim.v[2].uv.x) / C;

		float uvyA = (e0a * prim.v[0].uv.y + e1a * prim.v[1].uv.y + e2a * prim.v[2].uv.y) / C;
		float uvyB = (e0b * prim.v[0].uv.y + e1b * prim.v[1].uv.y + e2b * prim.v[2].uv.y) / C;
		float uvyD = (e0c * prim.v[0].uv.y + e1c * prim.v[1].uv.y + e2c * prim.v[2].uv.y) / C;

		data.oneOverW = glm::vec3(oneOverW_A, oneOverW_B, oneOverW_D);
		data.zPlaneEq = glm::vec3(zA, zB, zD);
		data.normalEqNx = glm::vec3(normNxA, normNxB, normNxD);
		data.normalEqNy = glm::vec3(normNyA, normNyB, normNyD);
		data.normalEqNz = glm::vec3(normNzA, normNzB, normNzD);
		data.uvEqx = glm::vec3(uvxA, uvxB, uvxD);
		data.uvEqy = glm::vec3(uvyA, uvyB, uvyD);
		data.edge[0] = glm::vec3(e0a, e0b, e0c);
		data.edge[1] = glm::vec3(e1a, e1b, e1c);
		data.edge[2] = glm::vec3(e2a, e2b, e2c);
		data.primitiveID = prim.primitiveID;

		triPreparedStream[idx] = data;
	}

	__device__ __forceinline__ float4 SetupTileCoverData(glm::vec3 e0, glm::vec3 e1, glm::vec3 e2)
	{
		unsigned flag = 0u;

		float tileMinX = blockIdx.x * BIN_PIXEL_SIZE;
		float tileMinY = blockIdx.y * BIN_PIXEL_SIZE;
		float tileMaxX = tileMinX + TILE_PIXEL_SIZE;
		float tileMaxY = tileMinY + TILE_PIXEL_SIZE;
		float e0disBase, e1disBase, e2disBase;
		{
			float chosenX = e0.x >= 0 ? tileMaxX : tileMinX;
			float chosenY = e0.y >= 0 ? tileMaxY : tileMinY;
			flag |= (e0.x >= 0) ? 0 : 1u << 0;
			flag |= (e0.y >= 0) ? 0 : 1u << 1;
			e0disBase = e0.x * chosenX + e0.y * chosenY + e0.z;
		}
		{
			float chosenX = e1.x >= 0 ? tileMaxX : tileMinX;
			float chosenY = e1.y >= 0 ? tileMaxY : tileMinY;
			flag |= (e1.x >= 0) ? 0 : 1u << 2;
			flag |= (e1.y >= 0) ? 0 : 1u << 3;
			e1disBase = e1.x * chosenX + e1.y * chosenY + e1.z;
		}
		{
			float chosenX = e2.x >= 0 ? tileMaxX : tileMinX;
			float chosenY = e2.y >= 0 ? tileMaxY : tileMinY;
			flag |= (e2.x >= 0) ? 0 : 1u << 4;
			flag |= (e2.y >= 0) ? 0 : 1u << 5;
			e2disBase = e2.x * chosenX + e2.y * chosenY + e2.z;
		}
		return make_float4(e0disBase, e1disBase, e2disBase, __uint_as_float(flag));
	}

	__global__ void CoarseRasterizer(const unsigned int* primitiveCount /*input a compacted primitive Size*/, const unsigned int* subQueueBaseIndex,
		const unsigned int* subQueuePrimCount, const unsigned* binQueue,
		const TriangleSetupData* triSetupData, unsigned* hiZ,
		unsigned* tileTrunkAllocator, unsigned* tileQueueBaseIndex, unsigned* tileQueuePrimCount, unsigned int* tileQueue,
		int width, int height)
	{
		//persistent thread
		// One block process one bin (256threads)
		// bin (128x128) -> tile (8x8)
		__shared__ unsigned sTriData[COARSE_RASTER_BLOCK_SIZE];
		__shared__ unsigned sHiZ[MAX_TILE_COUNT];
		__shared__ unsigned sTileMask[COARSE_RASTER_BLOCK_SIZE >> 5][MAX_TILE_COUNT + 1];
		__shared__ unsigned sTileTrunkCount[32];

		__shared__ unsigned int sFillCount;
		__shared__ unsigned int sToFill;
		__shared__ unsigned int sLastWriteOffset;
		__shared__ unsigned int sLastReadOffset;
		__shared__ unsigned int sSubQueueCount;
		__shared__ bool sNoMoreWork;

		__shared__ unsigned int sTrunkAllocationBase;

		//sub queue count = total primitive count / threads per block
		const int trianglePerSubQueue = 256;
		int subQueueCount = *primitiveCount;
		subQueueCount = (subQueueCount + trianglePerSubQueue - 1) / trianglePerSubQueue;

		int binIdx = blockIdx.x + blockIdx.y * BIN_PER_ROW;
		int tidx = threadIdx.x + threadIdx.y * TILE_PER_ROW;

		unsigned tileX = blockIdx.x * TILE_PER_ROW + threadIdx.x;
		unsigned tileY = blockIdx.y * TILE_PER_ROW + threadIdx.y;
		int tileIdx = tileX + tileY * TILE_PER_ROW * BIN_PER_ROW;

		unsigned lastTrunkRemain = 0;
		unsigned lastTrunkOffset = 0;
		unsigned tileQueueEntryCount = 0;
		int tileReadEntry = -1;

		//init hi-z cache
		sHiZ[tidx] = hiZ[binIdx * MAX_TILE_COUNT + tidx];

		unsigned intersectMask[MAX_TILE_COUNT >> 5];

		if (tidx == 0)
		{
			sLastWriteOffset = 0;
			sLastReadOffset = 0;
			sSubQueueCount = 0;
			sNoMoreWork = false;
			sTrunkAllocationBase = 0;
		}

		while (true)
		{
			if (tidx == 0)
			{
				sFillCount = 0;
				sToFill = 0;
			}
			__syncthreads();
			for (int i = 0; i < (MAX_TILE_COUNT >> 5); i++) intersectMask[i] = 0;

			while (sFillCount < COARSE_RASTER_BLOCK_SIZE)
			{
				if (tidx == 0)
				{
					if (sLastReadOffset > sLastWriteOffset) // read last
					{
						sToFill = min(sLastReadOffset - sLastWriteOffset, COARSE_RASTER_BLOCK_SIZE - sFillCount);
					}
					else // read new sub queue
					{
						sSubQueueCount++;
						if (sSubQueueCount > subQueueCount) //last subqueue
						{
							sNoMoreWork = true;
						}
						if (!sNoMoreWork)
						{
							sLastWriteOffset = subQueueBaseIndex[(sSubQueueCount - 1) * MAX_BIN_COUNT + binIdx];
							sLastReadOffset = sLastWriteOffset + subQueuePrimCount[(sSubQueueCount - 1) * MAX_BIN_COUNT + binIdx];
							sToFill = min(sLastReadOffset - sLastWriteOffset, COARSE_RASTER_BLOCK_SIZE - sFillCount);
						}
					}
				}
				__syncthreads();

				if (sNoMoreWork) break;

				if (tidx < sToFill)
				{
					unsigned int lookIDX = binQueue[sLastWriteOffset + tidx];
					sTriData[sFillCount + tidx] = lookIDX;
				}
				__syncthreads();
				if (tidx == 0)
				{
					sFillCount += sToFill;
					sLastWriteOffset += sToFill;
				}
				__syncthreads();
			}
			__syncthreads();

			if (sFillCount == 0)
			{
				break; // all task finished
			}

			if (tidx < sFillCount)
			{
				TriangleSetupData data = triSetupData[sTriData[tidx]];

				////set data for the base tile
				auto [e0disBase, e1disBase, e2disBase, flagF] = SetupTileCoverData(data.edge[0], data.edge[1], data.edge[2]);
				unsigned directionalFlag = __float_as_uint(flagF);

				for (int y = 0; y < TILE_PER_ROW; y++)
				{
					for (int x = 0; x < TILE_PER_ROW; x++)
					{

						float bdminx = blockIdx.x * BIN_PIXEL_SIZE + x * TILE_PIXEL_SIZE;
						float bdminy = blockIdx.y * BIN_PIXEL_SIZE + y * TILE_PIXEL_SIZE;
						float bdmaxx = bdminx + TILE_PIXEL_SIZE;
						float bdmaxy = bdminy + TILE_PIXEL_SIZE;

						// rough test with bounding box
						glm::vec4 bounding = Intersect(data.bounding, { bdminx,bdminy, bdmaxx,bdmaxy });
						if (bounding.x >= bounding.z || bounding.y >= bounding.w) continue;

						// fine test with edge function
						float e0dis = e0disBase + data.edge[0].x * (x * TILE_PIXEL_SIZE) + data.edge[0].y * (y * TILE_PIXEL_SIZE);
						float e1dis = e1disBase + data.edge[1].x * (x * TILE_PIXEL_SIZE) + data.edge[1].y * (y * TILE_PIXEL_SIZE);
						float e2dis = e2disBase + data.edge[2].x * (x * TILE_PIXEL_SIZE) + data.edge[2].y * (y * TILE_PIXEL_SIZE);
						if (e0dis < 0 || e1dis < 0 || e2dis < 0) continue;

						//Hi-z
					/*	float minXZa = bounding.x * data.zPlaneEq.x;
						float maxXZa = bounding.z * data.zPlaneEq.x;
						float minYZb = bounding.y * data.zPlaneEq.y;
						float maxYZb = bounding.w * data.zPlaneEq.y;*/
						float minXZa = bdminx * data.zPlaneEq.x;
						float maxXZa = bdmaxx * data.zPlaneEq.x;
						float minYZb = bdminy * data.zPlaneEq.y;
						float maxYZb = bdmaxy * data.zPlaneEq.y;
						float minZ = min(minXZa, maxXZa) + min(minYZb, maxYZb) + data.zPlaneEq.z;
						float maxZ = max(minXZa, maxXZa) + max(minYZb, maxYZb) + data.zPlaneEq.z;
						unsigned minZUint = NormToUnsigned_24Bit(minZ);
						unsigned maxZUint = NormToUnsigned_24Bit(maxZ);
						int tileIdx = x + y * TILE_PER_ROW;

						unsigned cachedZ = sHiZ[tileIdx];
						if (cachedZ <= minZUint) continue; //whole tile occluded

						//forward update hiz
						//as we seperate hiz and fine raster in different pass to decrease fine raster overhead of invisible tile
						// we only have a forward update for hiz here when the whole tile covered by tri
						// the feedback reduction from early-z is omited

						//if the triangle covers the whole tile
						int xInc = (directionalFlag & 0x1) ? 1 : -1;
						int yInc = (directionalFlag & 0x2) ? 1 : -1;
						float farPointE0 = e0disBase + data.edge[0].x * ((x + xInc) * TILE_PIXEL_SIZE) + data.edge[0].y * ((y + yInc) * TILE_PIXEL_SIZE);
						xInc = (directionalFlag & 0x4) ? 1 : -1;
						yInc = (directionalFlag & 0x8) ? 1 : -1;
						float farPointE1 = e1disBase + data.edge[1].x * ((x + xInc) * TILE_PIXEL_SIZE) + data.edge[1].y * ((y + yInc) * TILE_PIXEL_SIZE);
						xInc = (directionalFlag & 0x10) ? 1 : -1;
						yInc = (directionalFlag & 0x20) ? 1 : -1;
						float farPointE2 = e2disBase + data.edge[2].x * ((x + xInc) * TILE_PIXEL_SIZE) + data.edge[2].y * ((y + yInc) * TILE_PIXEL_SIZE);
						if (farPointE0 > 0.0f && farPointE1 > 0.0f && farPointE2 > 0.0f)
							atomicMin(&sHiZ[tileIdx], maxZUint);

						//set mask for the tile
						unsigned int mask = tileIdx & 0x1F;
						intersectMask[tileIdx >> 5] |= (1 << mask);
					}
				}
			}

			unsigned int warpMask = 0;
			for (int i = 0; i < MAX_TILE_COUNT; i++)
			{
				int laneCounter = i & 0x1F;
				bool predicate = (intersectMask[i >> 5] & (1u << laneCounter)) != 0;
				unsigned warpBallot = __ballot_sync(0xFFFFFFFF, predicate);
				if (laneCounter == get_lane_id()) warpMask = warpBallot;
				if (laneCounter == 31)
					sTileMask[tidx >> 5][i - 31 + get_lane_id()] = warpMask; //first use it as a mask storage
			}
			__syncthreads();

			for (int i = 0; i < MAX_TILE_COUNT >> 5; i++)
			{
				intersectMask[i] = sTileMask[i][tidx];
			}

			unsigned int sum = 0;
			for (int i = 0; i < MAX_TILE_COUNT >> 5; i++)
			{
				unsigned int mask = intersectMask[i];
				unsigned int warpSum = __popc(mask);
				sum += warpSum;
			}

			if (tidx < 32) sTileTrunkCount[tidx] = 0; //clean the shared buffer
			__syncthreads();
			unsigned int allocCount = max((int)sum - (int)lastTrunkRemain, 0);
			unsigned int tileTrunkCount = (allocCount + TILE_QUEUE_TRUNK_SIZE_UINT - 1) / TILE_QUEUE_TRUNK_SIZE_UINT;
			unsigned int trunkBase = ScanInBlockInclusive(tileTrunkCount, sTileTrunkCount);
			unsigned int laneId = get_lane_id();

			if (tidx == 255)
			{
				sTrunkAllocationBase = atomicAdd(tileTrunkAllocator, trunkBase);
			}
			__syncthreads();

			if (tileTrunkCount > 0) // need new trunk
			{
				tileQueueBaseIndex[tileQueueEntryCount + tileIdx * TILE_QUEUE_ENTRY] = (sTrunkAllocationBase + trunkBase - tileTrunkCount) * TILE_QUEUE_TRUNK_SIZE_UINT;
				tileQueuePrimCount[tileQueueEntryCount + tileIdx * TILE_QUEUE_ENTRY] = tileTrunkCount * TILE_QUEUE_TRUNK_SIZE_UINT;
				tileQueueEntryCount++;
			}

			for (int i = 0; i < COARSE_RASTER_BLOCK_SIZE; i++)
			{
				int laneCounter = i & 0x0000001F;
				bool predicate = (intersectMask[i >> 5] & (1u << laneCounter)) != 0;
				if (predicate) //need write
				{
					if (lastTrunkRemain == 0)  //switch to new trunk
					{
						tileReadEntry++;
						lastTrunkOffset = tileQueueBaseIndex[tileReadEntry + tileIdx * TILE_QUEUE_ENTRY];
						lastTrunkRemain = tileQueuePrimCount[tileReadEntry + tileIdx * TILE_QUEUE_ENTRY];
					}

					unsigned int globalIdx = lastTrunkOffset;
					tileQueue[globalIdx] = sTriData[i];

					lastTrunkOffset++;
					lastTrunkRemain--;
				}
			}
		}

		int tileIdxA = tileX + tileY * TILE_PER_ROW * BIN_PER_ROW;
		__syncthreads();
		if (tileQueueEntryCount != 0)
			tileQueuePrimCount[tileQueueEntryCount - 1 + tileIdxA * TILE_QUEUE_ENTRY] -= lastTrunkRemain;
		//write back hi-z cache into hi-z buffer
		hiZ[binIdx * MAX_TILE_COUNT + tidx] = sHiZ[tidx];
	}

	__device__ unsigned long long TileCoverageEachEdge(float2 baseXY, glm::vec3 edge)
	{
		unsigned long long coverage = 0ull;

		float dis00 = edge.x * baseXY.x + edge.y * baseXY.y + edge.z;
		float dis01 = dis00 + edge.x;
		float dis02 = dis01 + edge.x;
		float dis03 = dis02 + edge.x;
		float dis04 = dis03 + edge.x;
		float dis05 = dis04 + edge.x;
		float dis06 = dis05 + edge.x;
		float dis07 = dis06 + edge.x;
		coverage |= (dis00 >= 0 ? 1ull << 0 : 0ull);
		coverage |= (dis01 >= 0 ? 1ull << 1 : 0ull);
		coverage |= (dis02 >= 0 ? 1ull << 2 : 0ull);
		coverage |= (dis03 >= 0 ? 1ull << 3 : 0ull);
		coverage |= (dis04 >= 0 ? 1ull << 4 : 0ull);
		coverage |= (dis05 >= 0 ? 1ull << 5 : 0ull);
		coverage |= (dis06 >= 0 ? 1ull << 6 : 0ull);
		coverage |= (dis07 >= 0 ? 1ull << 7 : 0ull);

		for (int i = 1; i < 8; i++)
		{
			dis00 += edge.y;
			dis01 += edge.y;
			dis02 += edge.y;
			dis03 += edge.y;
			dis04 += edge.y;
			dis05 += edge.y;
			dis06 += edge.y;
			dis07 += edge.y;

			coverage |= (dis00 >= 0 ? 1ull << (i * 8 + 0) : 0ull);
			coverage |= (dis01 >= 0 ? 1ull << (i * 8 + 1) : 0ull);
			coverage |= (dis02 >= 0 ? 1ull << (i * 8 + 2) : 0ull);
			coverage |= (dis03 >= 0 ? 1ull << (i * 8 + 3) : 0ull);
			coverage |= (dis04 >= 0 ? 1ull << (i * 8 + 4) : 0ull);
			coverage |= (dis05 >= 0 ? 1ull << (i * 8 + 5) : 0ull);
			coverage |= (dis06 >= 0 ? 1ull << (i * 8 + 6) : 0ull);
			coverage |= (dis07 >= 0 ? 1ull << (i * 8 + 7) : 0ull);
		}
		return coverage;
	}

	__device__ unsigned long long TileCoverage(int2 tile, int width, int height, glm::vec3 edge0, glm::vec3 edge1, glm::vec3 edge2)
	{
		unsigned long long coverage = 0ull;
		float baseX = tile.x * TILE_PIXEL_SIZE + 0.5f;
		float baseY = tile.y * TILE_PIXEL_SIZE + 0.5f;

		unsigned long long cov0 = TileCoverageEachEdge(make_float2(baseX, baseY), edge0);
		unsigned long long cov1 = TileCoverageEachEdge(make_float2(baseX, baseY), edge1);
		unsigned long long cov2 = TileCoverageEachEdge(make_float2(baseX, baseY), edge2);

		unsigned long long boundMaskX = 0x000000FF >> max(tile.x * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE - width, 0);
		unsigned boundMaskY = 0x000000FF >> max(tile.y * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE - height, 0);
		unsigned long long boundMask =
			(boundMaskY & 1) * (boundMaskX << 0)
			| (boundMaskY & 2) * (boundMaskX << 7)
			| (boundMaskY & 4) * (boundMaskX << 14)
			| (boundMaskY & 8) * (boundMaskX << 21)
			| (boundMaskY & 16) * (boundMaskX << 28)
			| (boundMaskY & 32) * (boundMaskX << 35)
			| (boundMaskY & 64) * (boundMaskX << 42)
			| (boundMaskY & 128) * (boundMaskX << 49);

		coverage = cov1 & cov2 & cov0 & boundMask;
		return coverage;
	}

	//one thread - one primitive's fine raster in one tile(16quads)
	//one time reads 32 primitives from the tile queue

	// data flow from now on:
	// for every prim in every tile -> test valid quad(cover & early-z) -> write to fragmentStream
	// -> pixel shader output to outFragmentStream(only can change color and z?)
	// -> rop stage one, use screen size buffer to queue each quad with same pixel
	// -> rop stage two, sort each pixel's queue and rop(z-test, blend, stencil) -> output framebuffer

	__global__ void FineRasterizerWIP(const unsigned int* tileQueueBaseIndex, const unsigned int* tileQueuePrimCount, const unsigned int* tileQueue,
		const TriangleSetupData* triSetupData, unsigned* depthBuffer, unsigned* quadAllocator, FragmentPSin* fragmentStream, int width, int height)
	{
		const float DEPTH_EPSILON = 0.0001f;

		__shared__ unsigned int sDepthCache[FINE_RASTER_TILE_PER_BLOCK][TILE_PIXEL_SIZE * TILE_PIXEL_SIZE]; // 8x8 pixel, * block process tile
		__shared__ unsigned int sTriData[FINE_RASTER_TILE_PER_BLOCK][FINE_RASTER_PER_TILING_WORK_THREAD * 2]; // every warp one tile

		__shared__ unsigned int sTileTrunkBegin[TILE_QUEUE_ENTRY * FINE_RASTER_TILE_PER_BLOCK];
		__shared__ unsigned int sTileTrunkCount[TILE_QUEUE_ENTRY * FINE_RASTER_TILE_PER_BLOCK];

		__shared__ unsigned int sQuadAllocationBase[FINE_RASTER_TILE_PER_BLOCK];

		int tileX = blockIdx.x;
		int tileY = blockIdx.y * FINE_RASTER_TILE_PER_BLOCK + threadIdx.y;
		int tileIdx = tileX + tileY * BIN_PER_ROW * TILE_PER_ROW;

		int pixelXD = tileX * TILE_PIXEL_SIZE;
		int pixelYD = tileY * TILE_PIXEL_SIZE;
		pixelXD = pixelXD + (threadIdx.x & (TILE_PIXEL_SIZE - 1));
		pixelYD = pixelYD + (threadIdx.x >> TILE_PIXEL_SIZE_LOG2);
		//init z cache
		//each thread read 2 pixel
		sDepthCache[threadIdx.y][threadIdx.x] = InBuffer(pixelXD, pixelYD, width, height) ? depthBuffer[pixelXD + pixelYD * width] : 0xFFFFFFFF;
		sDepthCache[threadIdx.y][threadIdx.x + 32] = InBuffer(pixelXD, pixelYD + 4, width, height) ? depthBuffer[pixelXD + pixelYD * width + 4 * width] : 0xFFFFFFFF;

		if (get_lane_id() < TILE_QUEUE_ENTRY)
		{
			sTileTrunkBegin[get_lane_id() + threadIdx.y * TILE_QUEUE_ENTRY] = tileQueueBaseIndex[tileIdx * TILE_QUEUE_ENTRY + get_lane_id()];
			sTileTrunkCount[get_lane_id() + threadIdx.y * TILE_QUEUE_ENTRY] = tileQueuePrimCount[tileIdx * TILE_QUEUE_ENTRY + get_lane_id()];
		}
		__syncthreads();

		int readPos = 0;
		int writePos = 0;
		int currentTrunk = 0;
		int currentTrunkOffset = 0;

		while (true)
		{
			unsigned long long coverage = 0ull;
			unsigned quadCoverage = 0u;

			while ((writePos - readPos) < FINE_RASTER_PER_TILING_WORK_THREAD && currentTrunk < TILE_QUEUE_ENTRY)
			{
				//fetch primitive index
				int idx = currentTrunkOffset + threadIdx.x;
				if (idx + 1 > sTileTrunkCount[currentTrunk + threadIdx.y * TILE_QUEUE_ENTRY]) idx = -1;

				if (idx != -1)
				{
					unsigned int triIdx = tileQueue[sTileTrunkBegin[currentTrunk + threadIdx.y * TILE_QUEUE_ENTRY] + idx];
					unsigned int threadWritePos = (writePos + threadIdx.x) & 63;
					sTriData[threadIdx.y][threadWritePos] = triIdx;
				}

				bool predicate = (idx != -1);
				writePos += __popc(__ballot_sync(0xFFFFFFFF, predicate));
				currentTrunkOffset += __popc(__ballot_sync(0xFFFFFFFF, predicate));
				if (__any_sync(0xFFFFFFFF, !predicate))
				{
					currentTrunk++;
					currentTrunkOffset = 0;
				}
			}

			if (writePos == readPos) break; // all task finished

			// calculate thread read 
			int threadReadPos = readPos + threadIdx.x;
			bool predicate = (threadReadPos < writePos);
			threadReadPos &= 63;

			if (predicate)
			{
				//if (sTriData[threadIdx.x] > 5000) printf("fine errer");
				unsigned int setupDataIdx = sTriData[threadIdx.y][threadReadPos];
				TriangleSetupData data = triSetupData[sTriData[threadIdx.y][threadReadPos]];

				coverage = TileCoverage(
					make_int2(tileX, tileY),
					width, height,
					triSetupData[setupDataIdx].edge[0], triSetupData[setupDataIdx].edge[1], triSetupData[setupDataIdx].edge[2]
				);

				for (int i = 0; i < 64; i++)
				{
					if ((coverage & (1ull << i)) == 0) continue; // not covered

					// early-z test
					int localX = i & (TILE_PIXEL_SIZE - 1);
					int localY = i >> TILE_PIXEL_SIZE_LOG2;
					int pixelX = tileX * TILE_PIXEL_SIZE + localX;
					int pixelY = tileY * TILE_PIXEL_SIZE + localY;

					float z = triSetupData[setupDataIdx].zPlaneEq.x * (pixelX + 0.5f) + triSetupData[setupDataIdx].zPlaneEq.y * (pixelY + 0.5f) + triSetupData[setupDataIdx].zPlaneEq.z;
					unsigned z24Bit = NormToUnsigned_24Bit(z);
					unsigned zOld = atomicMin(&sDepthCache[threadIdx.y][localX + localY * TILE_PIXEL_SIZE], z24Bit);
					if (z24Bit > zOld) // early-z test fail
					{
						coverage &= ~(1ull << i);
					}
				}
			}
			readPos += __popc(__ballot_sync(0xFFFFFFFF, predicate));

			unsigned long long offsetMask = 0x303;
			for (int i = 0; i < 16; i++)
			{
				if ((i & 3) == 0 && i != 0) offsetMask <<= 8;
				unsigned long long quadMask = coverage & offsetMask;
				bool quadValid = __popcll(quadMask) > 0 ? true : false;
				quadCoverage |= quadValid ? 1u << i : 0u;

				offsetMask <<= 2;
			}

			//write fragment
			int writeFragCount = __popc(quadCoverage) * 4; //4 pixel per quad
			int laneWriteBaseIdx = ScanInWarpExclusive(writeFragCount);

			if (threadIdx.x == 31)
			{
				sQuadAllocationBase[threadIdx.y] = atomicAdd(quadAllocator, laneWriteBaseIdx + writeFragCount);
			}
			__syncwarp();

			//in a warp, every 4 threads process one quad (each thread write one pixel)
			unsigned predicateUnsigned = predicate ? 1 : 0;
			for (int i = 0; i < FINE_RASTER_PER_TILING_WORK_THREAD >> 3; i++)
			{
				int count = 0;

				int srcLane = (get_lane_id() >> 2) + 8 * i;
				unsigned validTriangle = __shfl_sync(0xFFFFFFFF, predicateUnsigned, srcLane, 32);
				int triReadPos = __shfl_sync(0xFFFFFFFF, threadReadPos, srcLane, 32);
				int writeBaseIdx = __shfl_sync(0xFFFFFFFF, laneWriteBaseIdx, srcLane, 32);
				unsigned triQuadCoverage = __shfl_sync(0xFFFFFFFF, quadCoverage, srcLane, 32);
				unsigned long long triCoverage = __shfl_sync(0xFFFFFFFF, coverage, srcLane, 32);

				if (validTriangle != 0)
				{
					TriangleSetupData data = triSetupData[sTriData[threadIdx.y][triReadPos & 63]];

					for (int j = 0; j < 16; j++)
					{
						int idInQuad = get_lane_id() & 3;
						int localX = (j & 0x3) * 2 + (idInQuad & 1);
						int localY = (j >> 2) * 2 + ((idInQuad >> 1) & 1);
						int pixelX = tileX * TILE_PIXEL_SIZE + localX;
						int pixelY = tileY * TILE_PIXEL_SIZE + localY;

						if ((triQuadCoverage & (1 << j)) == 0) continue;
						FragmentPSin fragment;

						//interpolate z and oneOverW
						float z0 = data.zPlaneEq.x * (pixelX + 0.5f) + data.zPlaneEq.y * (pixelY + 0.5f) + data.zPlaneEq.z - DEPTH_EPSILON;
						float oneOverW0 = data.oneOverW.x * (pixelX + 0.5f) + data.oneOverW.y * (pixelY + 0.5f) + data.oneOverW.z;
						float w0 = 1.0f / oneOverW0;
						float nx0 = data.normalEqNx.x * (pixelX + 0.5f) + data.normalEqNx.y * (pixelY + 0.5f) + data.normalEqNx.z;
						float ny0 = data.normalEqNy.x * (pixelX + 0.5f) + data.normalEqNy.y * (pixelY + 0.5f) + data.normalEqNy.z;
						float nz0 = data.normalEqNz.x * (pixelX + 0.5f) + data.normalEqNz.y * (pixelY + 0.5f) + data.normalEqNz.z;
						float uvx0 = data.uvEqx.x * (pixelX + 0.5f) + data.uvEqx.y * (pixelY + 0.5f) + data.uvEqx.z;
						float uvy0 = data.uvEqy.x * (pixelX + 0.5f) + data.uvEqy.y * (pixelY + 0.5f) + data.uvEqy.z;
						unsigned mask0 = ((triCoverage & (1ull << (localX + localY * 8))) > 0ull);
						//fragment.fragInfo = make_uint2(data.primitiveID, mask0);
						//fragment.uv = make_float2(uvx0 * w0, uvy0 * w0);
						fragment.packedData = make_uint4(data.primitiveID, mask0, __float_as_uint(uvx0 * w0), __float_as_uint(uvy0 * w0));
						fragment.sv_position = make_float4(pixelX, pixelY, __uint_as_float(NormToUnsigned_24Bit(z0)), w0);
						fragment.normal = make_float3(nx0 * w0, ny0 * w0, nz0 * w0);
						fragmentStream[sQuadAllocationBase[threadIdx.y] + writeBaseIdx + count + idInQuad] = fragment;

						count += 4;
					}
				}
			}
		}

		//write back hi-z cache into hi-z buffer
		if (InBuffer(pixelXD, pixelYD, width, height)) //each thread write 2 pixel
			depthBuffer[pixelXD + pixelYD * width] = sDepthCache[threadIdx.y][threadIdx.x];
		if (InBuffer(pixelXD, pixelYD + 4, width, height))
			depthBuffer[pixelXD + (pixelYD + 4) * width] = sDepthCache[threadIdx.y][threadIdx.x + 32];

	}

	__global__ void PixelShader(unsigned int* dimension, const FragmentPSin* fragmentStream, FragmentPSOut* outputStream, int* pixelBaseIdx, int width, int height, Texture2D tex)
	{
		int tidx = blockIdx.x * blockDim.x + threadIdx.x;
		if (tidx >= *dimension) return;

		//FragmentPSin input = fragmentStream[tidx];
		FragmentPSOut output;

		uint4 packedData = fragmentStream[tidx].packedData;
		float4 svPosition = fragmentStream[tidx].sv_position;
		float2 uv = make_float2(__uint_as_float(packedData.z), __uint_as_float(packedData.w));
		uint2 fragInfo = make_uint2(packedData.x, packedData.y);
		int x = svPosition.x;
		int y = svPosition.y;
		int pixel = x + y * width;
		unsigned depth = __float_as_uint(svPosition.z);

		//shading here
		float4 color = SampleTexture2D(tex, make_float2(uv.x, uv.y));

		output.color = make_float4(color.x, color.y, color.z, 1.0f);

		//gamma correction
		//output.color.r = powf(output.color.r, 1.0f / 2.2f);
		//	output.color.g = powf(output.color.g, 1.0f / 2.2f);
		//	output.color.b = powf(output.color.b, 1.0f / 2.2f);

		//link to pixel chain
		int nextFragmentIdx = atomicExch(&pixelBaseIdx[pixel], tidx);

		// pass pixel & quad info
		output.packedData = make_uint4(fragInfo.y, depth, fragInfo.x, nextFragmentIdx);

		outputStream[tidx] = output;
	}

	__global__ void ROP(const FragmentPSOut* fragmentStream, const int* pixelBaseIdx, int width, int height, unsigned* depthBuffer, float4* renderTarget)
	{
		const bool zWriteEnable = true;

		__shared__ int fragIndex[4][256];
		__shared__ int primitiveIndex[4][256];

		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;
		int tIdx = threadIdx.x + threadIdx.y * blockDim.x;

		if (x >= width || y >= height) return;

		int pixel = x + y * width;

		int fragIdx = pixelBaseIdx[pixel];
		//int fragIndex[4];
		//int primitiveIndex[4];

		int counter = 0;
		while (fragIdx > 0)
		{
			uint4 packedData = fragmentStream[fragIdx].packedData;
			if (packedData.x != 0)
			{
				fragIndex[counter][tIdx] = fragIdx;
				primitiveIndex[counter][tIdx] = packedData.z;
				++counter;
			}
			fragIdx = packedData.w;
		}

		//sort the fragment by primitiveID smaller primitiveID first
		for (int i = 1; i < counter; i++)
		{
			int primIdx = primitiveIndex[i][tIdx];
			int fragIdx = fragIndex[i][tIdx];
			// search for insert pos
			int start = 0;
			int end = i - 1;
			while (start < end)
			{
				int mid = (start + end) >> 1;
				if (primitiveIndex[mid][tIdx] < primIdx)
					start = mid + 1;
				else
					end = mid;
			}
			if (primitiveIndex[start][tIdx] < primIdx) start++;
			// move the data
			for (int j = i; j > start; j--)
			{
				primitiveIndex[j][tIdx] = primitiveIndex[j - 1][tIdx];
				fragIndex[j][tIdx] = fragIndex[j - 1][tIdx];
			}
			primitiveIndex[start][tIdx] = primIdx;
			fragIndex[start][tIdx] = fragIdx;
		}

		// do rop
		float4 dstColor = renderTarget[pixel];
		unsigned dstDepth = depthBuffer[pixel];

		for (int i = 0; i < counter; i++)
		{
			FragmentPSOut frag = fragmentStream[fragIndex[i][tIdx]];
			unsigned srcDepth = frag.packedData.y;
			unsigned mask = frag.packedData.x;
			// cover mask
			if (mask == 0) continue;
			//z-test
			if (srcDepth > dstDepth) continue;
			dstDepth = srcDepth;

			//blend
			float4 srcColor = frag.color;
			dstColor = srcColor;
		}

		renderTarget[pixel] = dstColor;
		if (zWriteEnable)
			depthBuffer[pixel] = dstDepth;

	}

	__global__ void StreamingToFrameBuffer(int dimension, float4* renderTarget, unsigned char* framebuffer, int width)
	{
		int tidx = blockIdx.x * blockDim.x + threadIdx.x;
		if (tidx >= dimension) return;

		float4 color = renderTarget[tidx];

		unsigned char r = (unsigned char)(glm::clamp(color.x, 0.0f, 1.0f) * 255.0f);
		unsigned char g = (unsigned char)(glm::clamp(color.y, 0.0f, 1.0f) * 255.0f);
		unsigned char b = (unsigned char)(glm::clamp(color.z, 0.0f, 1.0f) * 255.0f);
		unsigned char a = (unsigned char)(glm::clamp(color.w, 0.0f, 1.0f) * 255.0f);

		((uchar4*)framebuffer)[tidx] = make_uchar4(r, g, b, a);

		//framebuffer[tidx * 4 + 0] = r; // Red
		//framebuffer[tidx * 4 + 1] = g; // Green
		//framebuffer[tidx * 4 + 2] = b; // Blue
		//framebuffer[tidx * 4 + 3] = a; // Alpha
	}

}

using namespace CRPipeline;

void InitializeCudaRasterizer(int width, int height)
{
	windowHeight = height;
	windowWidth = width;

	CUDA_CHECK(cudaMalloc((void**)&dFragmentStream, sizeof(FragmentPSin) * windowHeight * windowWidth * AVERAGE_OVERDRAW));
	CUDA_CHECK(cudaMalloc((void**)&dFragmentOutStream, sizeof(FragmentPSOut) * windowHeight * windowWidth * AVERAGE_OVERDRAW));
	CUDA_CHECK(cudaMalloc((void**)&dRenderTarget, sizeof(float4) * windowHeight * windowWidth));
	CUDA_CHECK(cudaMalloc((void**)&dHiZ, sizeof(unsigned int) * MAX_BIN_COUNT * MAX_TILE_COUNT));

	CUDA_CHECK(cudaMalloc((void**)&dSubTriangleCounter, sizeof(unsigned int)));
	CUDA_CHECK(cudaMalloc((void**)&dTrunkAllocator, sizeof(unsigned int)));
	CUDA_CHECK(cudaMalloc((void**)&dTileTrunkAllocator, sizeof(unsigned int)));
	CUDA_CHECK(cudaMalloc((void**)&dQuadAllocator, sizeof(unsigned int)));
	CUDA_CHECK(cudaMalloc((void**)&dBinSubQueueCounter, sizeof(unsigned int)));
	CUDA_CHECK(cudaMalloc((void**)&dPrimitiveCounter, sizeof(unsigned int)));

	CUDA_CHECK(cudaMallocHost((void**)&hBinTrunkCount, sizeof(unsigned int)));
	CUDA_CHECK(cudaMallocHost((void**)&hTileTrunkCount, sizeof(unsigned int)));
	CUDA_CHECK(cudaMallocHost((void**)&hFragmentCount, sizeof(unsigned int)));
	CUDA_CHECK(cudaMallocHost((void**)&hdcPrimitiveCount, sizeof(unsigned int)));

	CUDA_CHECK(cudaMalloc((void**)&dSubQueueBaseIndex, sizeof(unsigned int) * MAX_BINNING_WAVE * BINNING_STAGE_BLOCK_SIZE));
	CUDA_CHECK(cudaMalloc((void**)&dSubQueuePrimCount, sizeof(unsigned int) * MAX_BINNING_WAVE * BINNING_STAGE_BLOCK_SIZE));
	CUDA_CHECK(cudaMalloc((void**)&dTileQueueBaseIndex, sizeof(unsigned) * MAX_BIN_COUNT * MAX_TILE_COUNT * TILE_QUEUE_ENTRY));
	CUDA_CHECK(cudaMalloc((void**)&dTileQueuePrimCount, sizeof(unsigned) * MAX_BIN_COUNT * MAX_TILE_COUNT * TILE_QUEUE_ENTRY));
	CUDA_CHECK(cudaMalloc((void**)&dTileQueue, sizeof(unsigned) * MAX_BIN_COUNT * MAX_TILE_COUNT * AVERAGE_TRIANGLE_PER_TILE));
	CUDA_CHECK(cudaMalloc((void**)&dPixelBaseIdx, sizeof(int) * windowWidth * windowHeight));

	InitCompactionEnvironment();
}

void RasterizerUpdateObjectsBuffer(int indexCountPerPrimitive, int vertexCount, int indexCount)
{
	if (dOutVertexStream) CUDA_CHECK(cudaFree(dOutVertexStream));
	if (dPrimitiveStream) CUDA_CHECK(cudaFree(dPrimitiveStream));
	if (dCompactedPrimitiveStream) CUDA_CHECK(cudaFree(dCompactedPrimitiveStream));
	if (dTriSetupData) CUDA_CHECK(cudaFree(dTriSetupData));

	//only support triangle as primitive now
	// so premitiveCount = indexCount / 3
	unsigned primitiveCount = (indexCount + indexCountPerPrimitive - 1) / indexCountPerPrimitive;

	CUDA_CHECK(cudaMalloc((void**)&dOutVertexStream, sizeof(VertexVSOut) * vertexCount));
	CUDA_CHECK(cudaMalloc((void**)&dPrimitiveStream, sizeof(Primitive) * primitiveCount * AVERAGE_PRIMITIVE_CULLED_COUNT)); // allocate 2 time for clipping case
	CUDA_CHECK(cudaMalloc((void**)&dCompactedPrimitiveStream, sizeof(Primitive) * primitiveCount * AVERAGE_PRIMITIVE_CULLED_COUNT >> 1));

	//set a minimum size for triangle setup data
	unsigned compactedCount = std::max(primitiveCount * AVERAGE_PRIMITIVE_CULLED_COUNT >> 1, 2048u);
	CUDA_CHECK(cudaMalloc((void**)&dTriSetupData, sizeof(TriangleSetupData) * compactedCount));
	size_t queueSize = sizeof(unsigned) * 128 * std::max(compactedCount, 256u);
	CUDA_CHECK(cudaMalloc((void**)&dBinQueue, sizeof(unsigned) * queueSize));
}

void CleanupCudaRasterizer()
{
	DestroyCompactionEnvironment();

	CUDA_CHECK(cudaFree(dFragmentStream));
	CUDA_CHECK(cudaFree(dFragmentOutStream));
	CUDA_CHECK(cudaFree(dRenderTarget));
	CUDA_CHECK(cudaFree(dHiZ));

	CUDA_CHECK(cudaFree(dSubTriangleCounter));
	CUDA_CHECK(cudaFree(dTrunkAllocator));
	CUDA_CHECK(cudaFree(dTileTrunkAllocator));
	CUDA_CHECK(cudaFree(dQuadAllocator));
	CUDA_CHECK(cudaFree(dBinSubQueueCounter));
	CUDA_CHECK(cudaFree(dPrimitiveCounter));

	CUDA_CHECK(cudaFreeHost(hBinTrunkCount));
	CUDA_CHECK(cudaFreeHost(hTileTrunkCount));
	CUDA_CHECK(cudaFreeHost(hFragmentCount));
	CUDA_CHECK(cudaFreeHost(hdcPrimitiveCount));

	CUDA_CHECK(cudaFree(dPrimitiveStream));
	CUDA_CHECK(cudaFree(dOutVertexStream));
	CUDA_CHECK(cudaFree(dCompactedPrimitiveStream));
	CUDA_CHECK(cudaFree(dTriSetupData));

	CUDA_CHECK(cudaFree(dSubQueueBaseIndex));
	CUDA_CHECK(cudaFree(dSubQueuePrimCount));
	CUDA_CHECK(cudaFree(dBinQueue));
	CUDA_CHECK(cudaFree(dTileQueueBaseIndex));
	CUDA_CHECK(cudaFree(dTileQueuePrimCount));
	CUDA_CHECK(cudaFree(dTileQueue));
	CUDA_CHECK(cudaFree(dPixelBaseIdx));
}

void BeginCudaFrame()
{
	// reset allocators
	CUDA_CHECK(cudaMemset(dTrunkAllocator, 0, sizeof(unsigned int)));
	CUDA_CHECK(cudaMemset(dTileTrunkAllocator, 0, sizeof(unsigned int)));
	CUDA_CHECK(cudaMemset(dQuadAllocator, 0, sizeof(unsigned int)));
	CUDA_CHECK(cudaMemset(dBinSubQueueCounter, 0, sizeof(unsigned int)));
	CUDA_CHECK(cudaMemcpy(dSubTriangleCounter, hdcPrimitiveCount, sizeof(unsigned int), cudaMemcpyHostToDevice));

	// clear depth buffer & render target & hiz
	CUDA_CHECK(cudaMemset(dFragmentStream, 0, sizeof(FragmentPSin) * windowWidth * windowHeight * AVERAGE_OVERDRAW));
	CUDA_CHECK(cudaMemset(dRenderTarget, 0, sizeof(float4) * windowHeight * windowWidth));
	CUDA_CHECK(cudaMemset(dHiZ, 0xFF, sizeof(unsigned) * MAX_BIN_COUNT * MAX_TILE_COUNT));
	CUDA_CHECK(cudaMemset(dDepthBuffer, 0xFF, sizeof(unsigned int) * windowWidth * windowHeight));

	//clear inner buffers
	CUDA_CHECK(cudaMemset(dPrimitiveStream, 0xFF, sizeof(Primitive) * dcPrimitiveCount * 4));
	//CUDA_CHECK(cudaMemset(dOutVertexStream, 0, sizeof(VertexVSOut) * dcVertexCount));
	CUDA_CHECK(cudaMemset(dPixelBaseIdx, 0xFF, sizeof(int) * windowWidth * windowHeight));

	CUDA_CHECK(cudaMemset(dSubQueuePrimCount, 0, sizeof(unsigned) * MAX_BINNING_WAVE * BINNING_STAGE_BLOCK_SIZE));
	CUDA_CHECK(cudaMemset(dTileQueuePrimCount, 0, sizeof(unsigned) * MAX_BIN_COUNT * MAX_TILE_COUNT * TILE_QUEUE_ENTRY));
}

void EndCudaFrame()
{
	CUDA_CHECK(cudaDeviceSynchronize());
}

void Rasterize(unsigned char* outRenderTarget, unsigned* depthBuffer,
	const VertexVSIn* vertexStream, const uint32_t* indexStream,
	int indexCount, int vertexCount, MatricesCBuffer* cb, Texture2D tex)
{
	const int threadsPerBlock = 256;

	//// set rtv & dsv & input stream & cbv ...

	SetGraphicsRoot(outRenderTarget, depthBuffer, vertexStream, indexStream, indexCount, vertexCount, cb, 0);
	BeginCudaFrame();

	// vertex fetch and shading
	{
		int blocksPerGrid = (dcVertexCount + threadsPerBlock - 1) / threadsPerBlock;
		VertexFetchAndShading << <blocksPerGrid, threadsPerBlock >> > (dcVertexCount, dInVertexStream, dOutVertexStream);
	}

	// primitive assembly
	{
		int blocksPerGrid = (dcPrimitiveCount + threadsPerBlock - 1) / threadsPerBlock;
		PrimitiveAssembly << <blocksPerGrid, threadsPerBlock >> > (dcPrimitiveCount, dIndexStream, dOutVertexStream, dPrimitiveStream, windowWidth, windowHeight, dSubTriangleCounter);
		PrimitiveCompaction(AVERAGE_PRIMITIVE_CULLED_COUNT * dcPrimitiveCount, dPrimitiveStream, dCompactedPrimitiveStream, dPrimitiveCounter, 0);
	}

	// rasterization
	{
		unsigned compactedCount = std::max(1u * dcPrimitiveCount * AVERAGE_PRIMITIVE_CULLED_COUNT >> 1, 2048u);
		int blocksPerGrid = (compactedCount + threadsPerBlock - 1) / threadsPerBlock;
		TriangleSetup << <blocksPerGrid, threadsPerBlock >> > (dPrimitiveCounter, dCompactedPrimitiveStream, dTriSetupData);
		PrimitiveBinning << <blocksPerGrid, threadsPerBlock >> > (dPrimitiveCounter, dCompactedPrimitiveStream, dTrunkAllocator, dSubQueueBaseIndex, dSubQueuePrimCount, dBinQueue, windowWidth, windowHeight);

		// one block = one sub queue --- subQueueCount = blocksPerGrid;
		// coarse raster only launch block in screen scissor plane
		// that is ceil(width / 128) * ceil(height / 128) blocks
		//tile queue index size  = 256 (bin) * 256 (tile) * TILE_QUEUE_ENTRY * sizeof uint
		//total queue size = 256 * 256 * 512(every tile contains 512 prim on average)
		int xUpper = UPPER_BOUND(windowWidth, BIN_PIXEL_SIZE_LOG2);
		int yUpper = UPPER_BOUND(windowHeight, BIN_PIXEL_SIZE_LOG2);
		dim3 blockSize(xUpper >> BIN_PIXEL_SIZE_LOG2, yUpper >> BIN_PIXEL_SIZE_LOG2);
		CoarseRasterizer << <blockSize, dim3(16, 16) >> > (dPrimitiveCounter/*divide by trianglePerSubQueue*/, dSubQueueBaseIndex, dSubQueuePrimCount, dBinQueue,
			dTriSetupData, dHiZ, dTileTrunkAllocator, dTileQueueBaseIndex, dTileQueuePrimCount, dTileQueue, windowWidth, windowHeight);

		int tileXUpper = UPPER_BOUND(windowWidth, TILE_PIXEL_SIZE_LOG2) >> TILE_PIXEL_SIZE_LOG2;
		int tileYUpper = UPPER_BOUND(windowHeight, TILE_PIXEL_SIZE_LOG2) >> TILE_PIXEL_SIZE_LOG2 >> 2;
		FineRasterizerWIP << <dim3(tileXUpper, tileYUpper), dim3(32, FINE_RASTER_TILE_PER_BLOCK) >> > (dTileQueueBaseIndex, dTileQueuePrimCount, dTileQueue,
			dTriSetupData, dDepthBuffer, dQuadAllocator, dFragmentStream, windowWidth, windowHeight);

		//CUDA_CHECK(cudaMemcpy(hFragmentCount, dQuadAllocator, sizeof(unsigned int), cudaMemcpyDeviceToHost));
	}

	// pixel shading & OM
	{
		int blocksPerGrid = (windowWidth * windowHeight * AVERAGE_OVERDRAW + threadsPerBlock - 1) / threadsPerBlock;
		PixelShader << <blocksPerGrid, threadsPerBlock >> > (dQuadAllocator, dFragmentStream, dFragmentOutStream, dPixelBaseIdx, windowWidth, windowHeight, tex);

		dim3 blockSize(16, 16);
		dim3 gridSize(
			(windowWidth + blockSize.x - 1) / blockSize.x,
			(windowHeight + blockSize.y - 1) / blockSize.y);
		ROP << <gridSize, blockSize >> > (dFragmentOutStream, dPixelBaseIdx, windowWidth, windowHeight, dDepthBuffer, dRenderTarget);
	}

	// output
	{
		int blocksPerGrid = (windowWidth * windowHeight + threadsPerBlock - 1) / threadsPerBlock;
		StreamingToFrameBuffer << <blocksPerGrid, threadsPerBlock >> > (windowWidth * windowHeight, dRenderTarget, dFrameBuffer, windowWidth);
	}

	EndCudaFrame();
}




