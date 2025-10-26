#ifndef RASTER_PIPE_HELPER_H
#define RASTER_PIPE_HELPER_H

#include "glm/glm/glm.hpp"

//#if defined(__CUDACC__)
//#else
//#include <cstdalign>
//struct alignas(16) float4 {
//	float x, y, z, w;
//};
//
//struct alignas(16) uint4 {
//	unsigned int x, y, z, w;
//};
//#endif

struct VertexVSIn
{
	glm::vec4 position; //POSITION0
	glm::vec3 normal; //NORMAL0
	glm::vec2 uv; //TEXCOORD0
};

struct VertexVSOut
{
	glm::vec4 sv_position; //SV_POSITION
	glm::vec3 normal; //NORMAL0
	glm::vec2 uv; //TEXCOORD0
};

struct MatricesCBuffer		//
{
	//DX mvp -> //perspectiveLH
	glm::mat4 mvp;	
};

struct Primitive
{
	VertexVSOut v[3];
	int primitiveID;
};

struct TriangleSetupData
{
	glm::vec4 bounding; //xy: min, zw: max
	glm::vec3 edge[3]; //edge0: 
	glm::vec3 zPlaneEq; 
	glm::vec3 oneOverW;
	int primitiveID;

	//other attributes interpolation
	glm::vec3 normalEqNx;
	glm::vec3 normalEqNy;
	glm::vec3 normalEqNz;

	glm::vec3 uvEqx;
	glm::vec3 uvEqy;
};

struct FragmentPSin
{
	//uint2 fragInfo; //x: primitiveID, y: mask
	uint4 packedData; //xy: fragInfo  zw: uv
	float4 sv_position; //xy: pixel center position, z: depth

	// other attributes interpolation
	float3 normal;
	//float2 uv;
};

struct FragmentPSOut
{
//	unsigned int mask;
//	unsigned int depth;
// 	int primitiveID;
//	int nextFragmentIdx;
	uint4 packedData; // packed data for variable above, for coalesced memory access
	float4 color;
};

struct Texture2D
{
	int width;
	int height;
	unsigned char* data; //RGBA8 
	//Default 4 level of mipmaps
	__host__ __device__ unsigned int getMipLevelBaseOffset(int mipLevel) const
	{
		unsigned int offset = 0;
		int w = width;
		int h = height;
		for (int i = 0; i < mipLevel; i++)
		{
			offset += w * h * 4;
			w = w >> 1;
			h = h >> 1;
		}
		return offset;
	}
};

struct SamplerState
{
};

#endif // !RASTER_PIPE_HELPER_H

