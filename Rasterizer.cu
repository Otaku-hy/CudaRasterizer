#include "Rasterizer.h"
#include "RasterUnitFunction.cuh"
#include "RasterParallelAlgorithm.cuh"



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


namespace  //// pipeline signature
{
	__constant__ MatricesCBuffer cbVertex;

	const VertexVSIn* dInVertexStream = nullptr;		// input vertex buffer
	VertexVSOut* dOutVertexStream = nullptr;	// output vertex buffer after vertex shader
	const uint32_t* dIndexStream = nullptr;				// index buffer
	Primitive* dPrimitiveStream = nullptr;		// primitive buffer after primitive assembly
	FragmentPSin* dFragmentStream = nullptr;	// fragment buffer after rasterization
	//float* dDepthBuffer = nullptr;			// depth buffer
	unsigned* dDepthBuffer = nullptr;
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
	unsigned int* dTrunkAllocator = nullptr;
	unsigned int* dSubQueueBaseIndex = nullptr;
	unsigned int* dSubQueuePrimCount = nullptr;
	unsigned int* dBinQueue = nullptr;
	TriangleSetupData* dTriSetupData = nullptr;
	unsigned int* dHiZ = nullptr;
	unsigned int* dTileTrunkAllocator = nullptr;
	unsigned int* dTileQueueBaseIndex = nullptr;
	unsigned int* dTileQueuePrimCount = nullptr;
	unsigned int* dTileQueue = nullptr;
	unsigned int* dQuadAllocator = nullptr;

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

	for (int j = 1; j < vertexCount - 1; j++)
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

__global__ void PrimitiveBinning(int size, const Primitive* primitiveStream, unsigned int* trunkAllocator, unsigned int* subQueueBaseIndex, unsigned int* subQueuePrimCount, unsigned int* queue, int width, int height)
{
	__shared__ unsigned sBinTriangleCountWarp[BINNING_STAGE_BLOCK_SIZE / 32 /*warp count*/][MAX_BIN_COUNT + 1];
	__shared__ unsigned sBinMask[BINNING_STAGE_BLOCK_SIZE / 32][MAX_BIN_COUNT + 1];
	__shared__ unsigned sBinTrunkCount[32];
	__shared__ unsigned int sTrunkAllocationBase;

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

__global__ void TriangleSetup(int size, Primitive* primitiveStream, TriangleSetupData* triPreparedStream)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= size) return;

	Primitive prim = primitiveStream[idx];

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

	data.oneOverW = glm::vec3(oneOverW_A, oneOverW_B, oneOverW_D);
	data.zPlaneEq = glm::vec3(zA, zB, zD);
	data.edge[0] = glm::vec3(e0a, e0b, e0c);
	data.edge[1] = glm::vec3(e1a, e1b, e1c);
	data.edge[2] = glm::vec3(e2a, e2b, e2c);
	data.primitiveID = prim.primitiveID;

	triPreparedStream[idx] = data;
}

__global__ void CoarseRasterizer(unsigned testSize, const int subQueueCount, const unsigned int* subQueueBaseIndex,
	const unsigned int* subQueuePrimCount, const unsigned* binQueue,
	const TriangleSetupData* triSetupData, unsigned* hiZ,
	unsigned* tileTrunkAllocator, unsigned* tileQueueBaseIndex, unsigned* tileQueuePrimCount, unsigned int* tileQueue,
	int width, int height, Primitive* pStream)
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

	int binIdx = blockIdx.x + blockIdx.y * BIN_PER_ROW;
	int tidx = threadIdx.x + threadIdx.y * TILE_PER_ROW;

	unsigned tileX = blockIdx.x * TILE_PER_ROW + threadIdx.x;
	unsigned tileY = blockIdx.y * TILE_PER_ROW + threadIdx.y;
	volatile int tileIdx = tileX + tileY * TILE_PER_ROW * BIN_PER_ROW;

	volatile unsigned lastTrunkRemain = 0;
	volatile unsigned lastTrunkOffset = 0;
	volatile unsigned tileQueueEntryCount = 0;
	volatile int tileReadEntry = -1;

	if (blockIdx.x == 0 && threadIdx.x == 0) pStream[0].primitiveID = 0;

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
						if (tileQueuePrimCount[(sSubQueueCount - 1) * MAX_BIN_COUNT + binIdx] > 0 && binQueue[sLastWriteOffset] > 5000)
							printf("fine error ");
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
			float tileMinX = blockIdx.x * BIN_PIXEL_SIZE;
			float tileMinY = blockIdx.y * BIN_PIXEL_SIZE;
			float tileMaxX = tileMinX + TILE_PIXEL_SIZE;
			float tileMaxY = tileMinY + TILE_PIXEL_SIZE;
			glm::vec4 tileBounding = { tileMinX,tileMinY, tileMaxX,tileMaxY };
			float e0disBase, e1disBase, e2disBase;
			{
				float chosenX = data.edge[0].x >= 0 ? tileMaxX : tileMinX;
				float chosenY = data.edge[0].y >= 0 ? tileMaxY : tileMinY;
				e0disBase = data.edge[0].x * chosenX + data.edge[0].y * chosenY + data.edge[0].z;
			}
			{
				float chosenX = data.edge[1].x >= 0 ? tileMaxX : tileMinX;
				float chosenY = data.edge[1].y >= 0 ? tileMaxY : tileMinY;
				e1disBase = data.edge[1].x * chosenX + data.edge[1].y * chosenY + data.edge[1].z;
			}
			{
				float chosenX = data.edge[2].x >= 0 ? tileMaxX : tileMinX;
				float chosenY = data.edge[2].y >= 0 ? tileMaxY : tileMinY;
				e2disBase = data.edge[2].x * chosenX + data.edge[2].y * chosenY + data.edge[2].z;
			}

			for (int y = 0; y < TILE_PER_ROW; y++)
			{
				for (int x = 0; x < TILE_PER_ROW; x++)
				{
					float bdminx = tileBounding.x + x * TILE_PIXEL_SIZE;
					float bdminy = tileBounding.y + y * TILE_PIXEL_SIZE;
					float bdmaxx = bdminx + TILE_PIXEL_SIZE;
					float bdmaxy = bdminy + TILE_PIXEL_SIZE;

					// rough test with bounding box

					glm::vec4 tileBounding = { bdminx,bdminy, bdmaxx,bdmaxy };
					glm::vec4 bounding = Intersect(data.bounding, tileBounding);
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
					if (cachedZ < minZUint) continue; //whole tile occluded

					//forward update hiz
					//as we seperate hiz and fine raster in different pass to decrease fine raster overhead of invisible tile
					// we only have a forward update for hiz here when the whole tile covered by tri
					// the feedback reduction from early-z is omited
					if (bounding, x == bdminx && bounding.y == bdminy && bounding.z == bdmaxx && bounding.w == bdmaxy)
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

		volatile unsigned int sum = 0;
		for (int i = 0; i < MAX_TILE_COUNT >> 5; i++)
		{
			unsigned int mask = intersectMask[i];
			unsigned int warpSum = __popc(mask);
			sum += warpSum;
		}

		if (tidx < 32) sTileTrunkCount[tidx] = 0; //clean the shared buffer
		__syncthreads();
		volatile unsigned int allocCount = max((int)sum - (int)lastTrunkRemain, 0);
		volatile unsigned int tileTrunkCount = (allocCount + TILE_QUEUE_TRUNK_SIZE_UINT - 1) / TILE_QUEUE_TRUNK_SIZE_UINT;
		volatile unsigned int trunkBase = ScanInBlockInclusive(tileTrunkCount, sTileTrunkCount);
		volatile unsigned int laneId = get_lane_id();

		if (tidx == 255)
		{
			sTrunkAllocationBase = atomicAdd(tileTrunkAllocator, trunkBase);
		}
		__syncthreads();

		if (tileTrunkCount > 0) // need new trunk
		{
			volatile unsigned aa = (sTrunkAllocationBase + trunkBase - tileTrunkCount) * TILE_QUEUE_TRUNK_SIZE_UINT;
			volatile unsigned bb = aa % 512;
			tileQueueBaseIndex[tileQueueEntryCount + tileIdx * TILE_QUEUE_ENTRY] = bb + aa;
			tileQueuePrimCount[tileQueueEntryCount + tileIdx * TILE_QUEUE_ENTRY] = tileTrunkCount * TILE_QUEUE_TRUNK_SIZE_UINT;
			tileQueueEntryCount++;
		}

		for (volatile int i = 0; i < COARSE_RASTER_BLOCK_SIZE; i++)
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

__global__ void FineRasterizer(const unsigned int* tileQueueBaseIndex, const unsigned int* tileQueuePrimCount, const unsigned int* tileQueue,
	const TriangleSetupData* triSetupData, unsigned* depthBuffer, unsigned* quadAllocator, FragmentPSin* fragmentStream, int width, int height, Primitive* primitiveStream)
{
	__shared__ unsigned int sDepthCache[TILE_PIXEL_SIZE * TILE_PIXEL_SIZE]; // 8x8 pixel, 
	__shared__ unsigned int sTriData[FINE_RASTER_BLOCK_SIZE];

	__shared__ unsigned int sFillCount;
	__shared__ unsigned int sToFill;
	__shared__ unsigned int sLastWriteOffset;
	__shared__ unsigned int sLastReadOffset;
	__shared__ unsigned int sSubQueueCount;
	__shared__ bool sNoMoreWork;

	__shared__ unsigned int sQuadAllocationBase;

	__shared__ unsigned long long test;

	int tileIdx = blockIdx.x + blockIdx.y * BIN_PER_ROW * TILE_PER_ROW;
	int pixelX = blockIdx.x * TILE_PIXEL_SIZE;
	int pixelY = blockIdx.y * TILE_PIXEL_SIZE;

	unsigned lastTrunkRemain = 0;
	unsigned lastTrunkOffset = 0;
	unsigned tileQueueCount = 0;

	pixelX = pixelX + threadIdx.x & (TILE_PIXEL_SIZE - 1);
	pixelY = pixelY + threadIdx.x >> TILE_PIXEL_SIZE_LOG2;
	//init z cache
	//each thread read 2 pixel
	sDepthCache[threadIdx.x] = InBuffer(pixelX, pixelY, width, height) ? depthBuffer[pixelX + pixelY * width] : 0xFFFFFFFF;
	sDepthCache[threadIdx.x + 32] = InBuffer(pixelX, pixelY + 4, width, height) ? depthBuffer[pixelX + pixelY * width + 4 * width] : 0xFFFFFFFF;

	if (threadIdx.x == 0)
	{
		sLastWriteOffset = 0;
		sLastReadOffset = 0;
		sSubQueueCount = 0;
		sNoMoreWork = false;
	}

	volatile unsigned long long coverage;
	unsigned quadCoverage;

	while (true)
	{
		if (threadIdx.x == 0)
		{
			sFillCount = 0;
			sToFill = 0;
		}
		coverage = 0ull;
		quadCoverage = 0u;

		__syncthreads();

		while (sFillCount < FINE_RASTER_BLOCK_SIZE)
		{
			if (threadIdx.x == 0)
			{
				if (sLastReadOffset > sLastWriteOffset) // read last
				{
					sToFill = min(sLastReadOffset - sLastWriteOffset, FINE_RASTER_BLOCK_SIZE - sFillCount);
				}
				else // read new sub queue
				{
					sSubQueueCount++;
					if (sSubQueueCount > TILE_QUEUE_ENTRY) //last subqueue
					{
						sNoMoreWork = true;
					}
					if (!sNoMoreWork)
					{
						sLastWriteOffset = tileQueueBaseIndex[tileIdx * TILE_QUEUE_ENTRY + sSubQueueCount - 1];
						sLastReadOffset = sLastWriteOffset + tileQueuePrimCount[tileIdx * TILE_QUEUE_ENTRY + sSubQueueCount - 1];
						sToFill = min(sLastReadOffset - sLastWriteOffset, FINE_RASTER_BLOCK_SIZE - sFillCount);
						if (tileQueuePrimCount[tileIdx * TILE_QUEUE_ENTRY + sSubQueueCount - 1] > 0 && tileQueue[sLastWriteOffset] > 5000)
							printf("fine error ");
					}
				}
			}
			__syncthreads();

			if (sNoMoreWork) break;

			if (threadIdx.x < sToFill)
			{
				volatile unsigned data1 = tileQueue[sLastWriteOffset + threadIdx.x];
				sTriData[sFillCount + threadIdx.x] = data1;
			}
			__syncthreads();
			if (threadIdx.x == 0)
			{
				sFillCount += sToFill;
				sLastWriteOffset += sToFill;
			}
			__syncthreads();
		}
		__syncthreads();

		if (sFillCount == 0) break; // all task finished

		if (threadIdx.x < sFillCount)
		{
			//if (sTriData[threadIdx.x] > 5000) printf("fine errer");
			TriangleSetupData data = triSetupData[sTriData[threadIdx.x]];

			coverage = TileCoverage(
				make_int2(blockIdx.x, blockIdx.y),
				width, height,
				data.edge[0], data.edge[1], data.edge[2]
			);

			for (int i = 0; i < 64; i++)
			{
				if ((coverage & (1ull << i)) == 0) continue; // not covered

				// early-z test
				int localX = i & (TILE_PIXEL_SIZE - 1);
				int localY = i >> TILE_PIXEL_SIZE_LOG2;
				int pixelX = blockIdx.x * TILE_PIXEL_SIZE + localX;
				int pixelY = blockIdx.y * TILE_PIXEL_SIZE + localY;

				float z = data.zPlaneEq.x * (pixelX + 0.5f) + data.zPlaneEq.y * (pixelY + 0.5f) + data.zPlaneEq.z;
				unsigned z24Bit = NormToUnsigned_24Bit(z);
				unsigned zOld = atomicMin(&sDepthCache[localX + localY * TILE_PIXEL_SIZE], z24Bit);
				if (z24Bit <= zOld) // early-z test success
					coverage |= (1ull << i);
			}
		}

		//coverage = 0xFFFFFFFFFFFFFFFFull;
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
		int writeCount = __popc(quadCoverage) * 4; //4 pixel per quad
		int laneWriteBaseIdx = ScanInWarpExclusive(writeCount);

		if (threadIdx.x == 31)
		{
			sQuadAllocationBase = atomicAdd(quadAllocator, laneWriteBaseIdx + writeCount);
		}
		__syncthreads();

		int count = 0;
		if (threadIdx.x < sFillCount)
		{
			TriangleSetupData data = triSetupData[sTriData[threadIdx.x]];
			for (int i = 0; i < 16; i++)
			{
				if ((quadCoverage & (1 << i)) == 0) continue;

				FragmentPSin fragment[4];

				int localX = (i & 0x3) * 2;
				int localY = (i >> 2) * 2;
				int pixelX = blockIdx.x * TILE_PIXEL_SIZE + localX;
				int pixelY = blockIdx.y * TILE_PIXEL_SIZE + localY;

				fragment[0].primitiveID = data.primitiveID;
				fragment[1].primitiveID = data.primitiveID;
				fragment[2].primitiveID = data.primitiveID;
				fragment[3].primitiveID = data.primitiveID;

				fragment[0].mask = ((coverage & (1ull << (localX + localY * 8))) > 0ull);
				fragment[1].mask = ((coverage & (1ull << (localX + 1 + localY * 8))) > 0ull);
				fragment[2].mask = ((coverage & (1ull << (localX + (localY + 1) * 8))) > 0ull);
				fragment[3].mask = ((coverage & (1ull << (localX + 1 + (localY + 1) * 8))) > 0ull);

				//interpolate z and oneOverW
				float z0 = data.zPlaneEq.x * (pixelX + 0.5f) + data.zPlaneEq.y * (pixelY + 0.5f) + data.zPlaneEq.z;
				float oneOverW0 = data.oneOverW.x * (pixelX + 0.5f) + data.oneOverW.y * (pixelY + 0.5f) + data.oneOverW.z;
				float w0 = 1.0f / oneOverW0;

				fragment[0].sv_position = glm::vec4(pixelX, pixelY, NormToUnsigned_24Bit(z0), w0);

				float z1 = z0 + data.zPlaneEq.x;
				float oneOverW1 = oneOverW0 + data.oneOverW.x;
				float w1 = 1.0f / oneOverW1;
				fragment[1].sv_position = glm::vec4(pixelX + 1, pixelY, NormToUnsigned_24Bit(z1), w1);

				float z2 = z0 + data.zPlaneEq.y;
				float oneOverW2 = oneOverW0 + data.oneOverW.y;
				float w2 = 1.0f / oneOverW2;
				fragment[2].sv_position = glm::vec4(pixelX, pixelY + 1, NormToUnsigned_24Bit(z2), w2);

				float z3 = z2 + data.zPlaneEq.x;
				float oneOverW3 = oneOverW2 + data.oneOverW.x;
				float w3 = 1.0f / oneOverW3;
				fragment[3].sv_position = glm::vec4(pixelX + 1, pixelY + 1, NormToUnsigned_24Bit(z3), w3);

				fragmentStream[sQuadAllocationBase + laneWriteBaseIdx + count + 0] = fragment[0];
				fragmentStream[sQuadAllocationBase + laneWriteBaseIdx + count + 1] = fragment[1];
				fragmentStream[sQuadAllocationBase + laneWriteBaseIdx + count + 2] = fragment[2];
				fragmentStream[sQuadAllocationBase + laneWriteBaseIdx + count + 3] = fragment[3];

				count += 4;
			}
		}
	}

	//write back hi-z cache into hi-z buffer
	if (InBuffer(pixelX, pixelY, width, height)) //each thread write 2 pixel
		depthBuffer[pixelX + pixelY * width] = sDepthCache[threadIdx.x];
	if (InBuffer(pixelX, pixelY + 4, width, height))
		depthBuffer[pixelX + (pixelY + 4) * width] = sDepthCache[threadIdx.x + 32];

}

__global__ void PixelShader(int dimension, const FragmentPSin* fragmentStream, const FragmentPSOut* renderTarget, int width, int height)
{
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (tidx >= dimension) return;

	FragmentPSin input = fragmentStream[tidx];
	int x = input.sv_position.x;
	int y = input.sv_position.y;

	int pixelIndex = y * width + x;

	if (input.mask == 1)
	{
		renderTarget[pixelIndex].color = glm::vec4(1, 1, 1, 1);
		renderTarget[pixelIndex].
	}

		renderTarget[pixelIndex] = glm::vec4(1, 1, 1, 1);

}

//__global__ void PixelShader(int dimension, FragmentPSin* fragmentStream, glm::vec4* renderTarget, int width, int height)
//{
//	int x = blockIdx.x * blockDim.x + threadIdx.x;
//	int y = blockIdx.y * blockDim.y + threadIdx.y;
//
//	if (x >= width || y >= height) return;
//
//	int pixelIndex = y * width + x;
//
//	FragmentPSin fragment = fragmentStream[pixelIndex];
//	renderTarget[pixelIndex] = fragment.color;
//}

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

	CUDA_CHECK(cudaMalloc((void**)&dFragmentStream, sizeof(FragmentPSin) * windowHeight * windowWidth * AVERAGE_OVERDRAW));
	CUDA_CHECK(cudaMalloc((void**)&dRenderTarget, sizeof(glm::vec4) * windowHeight * windowWidth));

	CUDA_CHECK(cudaMalloc((void**)&dSubTriangleCounter, sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&dTrunkAllocator, sizeof(unsigned int)));
	CUDA_CHECK(cudaMalloc((void**)&dTileTrunkAllocator, sizeof(unsigned int)));
	CUDA_CHECK(cudaMalloc((void**)&dQuadAllocator, sizeof(unsigned int)));

	CUDA_CHECK(cudaMalloc((void**)&dHiZ, sizeof(unsigned int) * MAX_BIN_COUNT * MAX_TILE_COUNT));

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
	cudaFree(dHiZ);

	cudaFree(dTrunkAllocator);
	cudaFree(dTileTrunkAllocator);
	cudaFree(dQuadAllocator);
}

void Rasterize(unsigned char* outRenderTarget, unsigned* depthBuffer,
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
	CUDA_CHECK(cudaMemset(dTrunkAllocator, 0, sizeof(unsigned int)));
	CUDA_CHECK(cudaMemset(dTileTrunkAllocator, 0, sizeof(unsigned int)));
	CUDA_CHECK(cudaMemset(dQuadAllocator, 0, sizeof(unsigned int)));
	CUDA_CHECK(cudaMemcpyToSymbol(cbVertex, cb, sizeof(MatricesCBuffer)));

	int threadsPerBlock = 256;

	// clear depth buffer & render target & hiz
	CUDA_CHECK(cudaMemset(dFragmentStream, 0, sizeof(FragmentPSin) * windowWidth * windowHeight * AVERAGE_OVERDRAW));
	//ClearDepthBuffer << <(windowWidth * windowHeight + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (windowWidth * windowHeight, dDepthBuffer);
	CUDA_CHECK(cudaMemset(dRenderTarget, 0, sizeof(glm::vec4) * windowHeight * windowWidth));
	CUDA_CHECK(cudaMemset(dHiZ, 0xFF, sizeof(unsigned) * MAX_BIN_COUNT * MAX_TILE_COUNT));
	CUDA_CHECK(cudaMemset(dDepthBuffer, 0xFF, sizeof(unsigned int) * windowWidth * windowHeight));

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
		CUDA_CHECK(cudaDeviceSynchronize());
	}

	// rasterization
	unsigned int fragmentCount = 0;
	if (primitiveCount != 0)
	{
		int blocksPerGrid = (primitiveCount + threadsPerBlock - 1) / threadsPerBlock;
		CUDA_CHECK(cudaMalloc((void**)&dTriSetupData, sizeof(TriangleSetupData) * primitiveCount));
		TriangleSetup << <blocksPerGrid, threadsPerBlock >> > (primitiveCount, dCompactedPrimitiveStream, dTriSetupData);
		CUDA_CHECK(cudaDeviceSynchronize());

		blocksPerGrid = (primitiveCount + threadsPerBlock - 1) / threadsPerBlock;
		size_t queueSize = sizeof(unsigned) * 64 * primitiveCount * 2;
		queueSize = std::max(queueSize, 1024 * 128ull);
		CUDA_CHECK(cudaMalloc((void**)&dBinQueue, queueSize));
		CUDA_CHECK(cudaMalloc((void**)&dSubQueueBaseIndex, sizeof(unsigned int) * blocksPerGrid * 256));
		CUDA_CHECK(cudaMalloc((void**)&dSubQueuePrimCount, sizeof(unsigned int) * blocksPerGrid * 256));
		CUDA_CHECK(cudaMemset(dSubQueuePrimCount, 0, sizeof(unsigned int) * 256 * blocksPerGrid));
		PrimitiveBinning << <blocksPerGrid, threadsPerBlock >> > (primitiveCount, dCompactedPrimitiveStream, dTrunkAllocator, dSubQueueBaseIndex, dSubQueuePrimCount, dBinQueue, windowWidth, windowHeight);
		CUDA_CHECK(cudaDeviceSynchronize());

		// one block = one sub queue
		int subQueueCount = blocksPerGrid;
		// coarse raster only launch block in screen scissor plane
		// that is ceil(width / 128) * ceil(height / 128) blocks
		//tile queue size index = 256 (bin) * 256 (tile) * TILE_QUEUE_ENTRY * sizeof uint
		//total queue size = (sizeof(unsigned int) * 32 * primitiveCount * 32 max: 256 * 256 * 256(every tile contains 256 prim
		size_t size = std::min(2048 * 8 * primitiveCount * (int)sizeof(unsigned), 1024 * 1024 * 16 * (int)sizeof(unsigned));
		size = std::max(size, 256ull * 256 * 256 * 2);
		CUDA_CHECK(cudaMalloc((void**)&dTileQueue, size));
		CUDA_CHECK(cudaMalloc((void**)&dTileQueueBaseIndex, MAX_BIN_COUNT * MAX_TILE_COUNT * TILE_QUEUE_ENTRY * sizeof(unsigned)));
		CUDA_CHECK(cudaMalloc((void**)&dTileQueuePrimCount, MAX_BIN_COUNT * MAX_TILE_COUNT * TILE_QUEUE_ENTRY * sizeof(unsigned)));
		CUDA_CHECK(cudaMemset(dTileQueuePrimCount, 0, MAX_BIN_COUNT * MAX_TILE_COUNT * TILE_QUEUE_ENTRY * sizeof(unsigned)));

		int xUpper = UPPER_BOUND(windowWidth, BIN_PIXEL_SIZE_LOG2);
		int yUpper = UPPER_BOUND(windowHeight, BIN_PIXEL_SIZE_LOG2);
		dim3 blockSize(xUpper >> BIN_PIXEL_SIZE_LOG2, yUpper >> BIN_PIXEL_SIZE_LOG2);
		CoarseRasterizer << <blockSize, dim3(16, 16) >> > (size >> 2, subQueueCount, dSubQueueBaseIndex, dSubQueuePrimCount, dBinQueue,
			dTriSetupData, dHiZ, dTileTrunkAllocator, dTileQueueBaseIndex, dTileQueuePrimCount, dTileQueue, windowWidth, windowHeight, dCompactedPrimitiveStream);
		CUDA_CHECK(cudaDeviceSynchronize());

		int tileXUpper = UPPER_BOUND(windowWidth, TILE_PIXEL_SIZE_LOG2) >> TILE_PIXEL_SIZE_LOG2;
		int tileYUpper = UPPER_BOUND(windowHeight, TILE_PIXEL_SIZE_LOG2) >> TILE_PIXEL_SIZE_LOG2;
		FineRasterizer << <dim3(tileXUpper, tileYUpper), 32 >> > (dTileQueueBaseIndex, dTileQueuePrimCount, dTileQueue,
			dTriSetupData, dDepthBuffer, dQuadAllocator, dFragmentStream, windowWidth, windowHeight, dCompactedPrimitiveStream);

		CUDA_CHECK(cudaMemcpy(&fragmentCount, dQuadAllocator, sizeof(unsigned int), cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaDeviceSynchronize());
	}

	if (fragmentCount != 0)
	{
		int blocksPerGrid = (fragmentCount + threadsPerBlock - 1) / threadsPerBlock;
		PixelShader << <blocksPerGrid, threadsPerBlock >> > (fragmentCount, dFragmentStream, dRenderTarget, windowWidth, windowHeight);
		CUDA_CHECK(cudaDeviceSynchronize());
	}

	//if (primitiveCount != 0)
	//{
	//	int blocksPerGrid = (primitiveCount + threadsPerBlock - 1) / threadsPerBlock;
	//	Rasterization << <blocksPerGrid, threadsPerBlock >> > (primitiveCount, dCompactedPrimitiveStream, dFragmentStream, dDepthBuffer, windowWidth, windowHeight);
	//	CUDA_CHECK(cudaGetLastError());
	//}

	// pixel shader
	//{
	//	dim3 blockSize(16, 16);
	//	dim3 gridSize(
	//		(windowWidth + blockSize.x - 1) / blockSize.x,
	//		(windowHeight + blockSize.y - 1) / blockSize.y);
	//	PixelShader << <gridSize, blockSize >> > (windowWidth * windowHeight, dFragmentStream, dRenderTarget, windowWidth, windowHeight);
	//}

	// output merger
	{
		int blocksPerGrid = (windowWidth * windowHeight + threadsPerBlock - 1) / threadsPerBlock;
		OutputMerger << <blocksPerGrid, threadsPerBlock >> > (windowWidth * windowHeight, dRenderTarget, dFrameBuffer, windowWidth);
		CUDA_CHECK(cudaDeviceSynchronize());
	}

	if (dBinQueue)
		CUDA_CHECK(cudaFree(dBinQueue));
	if (dSubQueueBaseIndex)
		CUDA_CHECK(cudaFree(dSubQueueBaseIndex));
	if (dSubQueuePrimCount)
		CUDA_CHECK(cudaFree(dSubQueuePrimCount));
	if (dTriSetupData)
		CUDA_CHECK(cudaFree(dTriSetupData));
	if (dTileQueue)
		CUDA_CHECK(cudaFree(dTileQueue));
	if (dTileQueueBaseIndex)
		CUDA_CHECK(cudaFree(dTileQueueBaseIndex));
	if (dTileQueuePrimCount)
		CUDA_CHECK(cudaFree(dTileQueuePrimCount));

	dBinQueue = nullptr;
	dSubQueueBaseIndex = nullptr;
	dSubQueuePrimCount = nullptr;
	dTriSetupData = nullptr;
	dTileQueue = nullptr;
	dTileQueueBaseIndex = nullptr;
	dTileQueuePrimCount = nullptr;
}




