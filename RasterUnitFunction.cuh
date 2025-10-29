#ifndef RASTER_FUNC
#define RASTER_FUNC

#include <cuda_runtime.h>

#include "RasterPipeHelper.h"

inline __device__ float atomicMinFloat(float* address, float value)
{
	unsigned int* address_as_ui = (unsigned int*)address;
	float ret = *address;
	while (value < ret)
	{
		float assume = ret;
		ret = __uint_as_float(atomicCAS(address_as_ui, __float_as_uint(assume), __float_as_uint(value)));
	}

	return ret;
}

inline __device__ unsigned int get_lane_id()
{
	unsigned int ret;
	asm("mov.u32 %0, %laneid;" : "=r"(ret));
	return ret;
}

inline __device__ bool InBuffer(int x, int y, int width, int height)
{
	return x >= 0 && x < width && y >= 0 && y < height;
}

inline __device__ float fracf(float val)
{
	return val - floorf(val);
}

template<typename T>
inline T __device__ ddx(T val)
{
	T other = __shfl_xor_sync(0xffffffff, val, 0x1, 4);
	if (get_lane_id() % 2) {
		return other - val;
	}
	else {
		return val - other;
	}
}

template<typename T>
inline T __device__ ddy(T val)
{
	T other = __shfl_xor_sync(0xffffffff, val, 0x2, 4);
	if (get_lane_id()  % 2) {
		return other - val;
	}
	else {
		return val - other;
	}
}

inline __device__ float4 FetchTexel(const unsigned char* tex, int idx)
{
	float4 result;
	uchar4 data = ((uchar4*)tex)[idx];
	//read
	result.x = data.x / 255.0f;
	result.y = data.y / 255.0f;
	result.z = data.z / 255.0f;
	result.w = data.w / 255.0f;
	return result;
}

inline __device__ float4 SampleTexture2DLOD(const Texture2D tex, float2 uv, int lod)
{
	lod = min(lod, 3); //clamp to max mip level

	float4 result;

	float texelSizeX = 1.0f / tex.width;
	float texelSizeY = 1.0f / tex.height;

	int width = tex.width >> lod;
	int height = tex.height >> lod;

	// warp mode : repeat
	float u = fracf(uv.x);
	float v = fracf(uv.y);
	v = 1.0f - v; //invert v

	//filter mode: nearest
	//int texU = roundf(u * tex.width);
	//int texV = roundf(v * tex.height);
	//float4 color00 = FetchTexel(tex, texU + texV * tex.width);

	//filter mode: bilinear
	int u0 = floorf(u * width);
	int v0 = floorf(v * height);
	int u1 = floorf(fracf(u + texelSizeX) * width);
	int v1 = floorf(fracf(v + texelSizeY) * height);
	unsigned base = tex.getMipLevelBaseOffset(lod);
	const unsigned char* texLod = &(tex.data[base]);
	float4 color00 = FetchTexel(texLod, u0 + v0 * width);
	float4 color10 = FetchTexel(texLod, u1 + v0 * width);
	float4 color01 = FetchTexel(texLod, u0 + v1 * width);
	float4 color11 = FetchTexel(texLod, u1 + v1 * width);

	float factorU = fracf(u * width - u0);
	float factorV = fracf(v * height - v0);
	color00.x = color00.x * (1.0f - factorU) + color10.x * factorU;
	color00.y = color00.y * (1.0f - factorU) + color10.y * factorU;
	color00.z = color00.z * (1.0f - factorU) + color10.z * factorU;
	color00.w = color00.w * (1.0f - factorU) + color10.w * factorU;

	color01.x = color01.x * (1.0f - factorU) + color11.x * factorU;
	color01.y = color01.y * (1.0f - factorU) + color11.y * factorU;
	color01.z = color01.z * (1.0f - factorU) + color11.z * factorU;
	color01.w = color01.w * (1.0f - factorU) + color11.w * factorU;

	result.x = color00.x * (1.0f - factorV) + color01.x * factorV;
	result.y = color00.y * (1.0f - factorV) + color01.y * factorV;
	result.z = color00.z * (1.0f - factorV) + color01.z * factorV;
	result.w = color00.w * (1.0f - factorV) + color01.w * factorV;

	return result;
}


inline __device__  float4 SampleTexture2D(const Texture2D tex, float2 uv)
{
	float4 result;

	//float texelSizeX = 1.0f / tex.width;
	//float texelSizeY = 1.0f / tex.height;

	// warp mode : repeat
	//float u = fracf(uv.x);
	//float v = fracf(uv.y);
	//v = 1.0f - v; //invert v

	//filter mode: nearest
	//int texU = roundf(u * tex.width);
	//int texV = roundf(v * tex.height);
	//float4 color00 = FetchTexel(tex, texU + texV * tex.width);

	//filter mode: bilinear
	//int u0 = floorf(u * tex.width);
	//int v0 = floorf(v * tex.height);
	//int u1 = floorf(fracf(u + texelSizeX) * tex.width);
	//int v1 = floorf(fracf(v + texelSizeY) * tex.height);
	//float4 color00 = FetchTexel(tex.data[0], u0 + v0 * tex.width);
	//float4 color10 = FetchTexel(tex.data[0], u1 + v0 * tex.width);
	//float4 color01 = FetchTexel(tex.data[0], u0 + v1 * tex.width);
	//float4 color11 = FetchTexel(tex.data[0], u1 + v1 * tex.width);

	//float factorU = fracf(u * tex.width - u0);
	//float factorV = fracf(v * tex.height - v0);
	//color00.x = color00.x * (1.0f - factorU) + color10.x * factorU;
	//color00.y = color00.y * (1.0f - factorU) + color10.y * factorU;
	//color00.z = color00.z * (1.0f - factorU) + color10.z * factorU;
	//color00.w = color00.w * (1.0f - factorU) + color10.w * factorU;

	//color01.x = color01.x * (1.0f - factorU) + color11.x * factorU;
	//color01.y = color01.y * (1.0f - factorU) + color11.y * factorU;
	//color01.z = color01.z * (1.0f - factorU) + color11.z * factorU;
	//color01.w = color01.w * (1.0f - factorU) + color11.w * factorU;

	//result.x = color00.x * (1.0f - factorV) + color01.x * factorV;
	//result.y = color00.y * (1.0f - factorV) + color01.y * factorV;
	//result.z = color00.z * (1.0f - factorV) + color01.z * factorV;
	//result.w = color00.w * (1.0f - factorV) + color01.w * factorV;

	//filter mode: trilinear
	glm::vec2 dx = { ddx(uv.x * tex.width) , ddx(uv.y * tex.height)};
	glm::vec2 dy = { ddy(uv.x * tex.width) , ddy(uv.y * tex.height) };
	float d = max(glm::dot(dx, dx), glm::dot(dy, dy));
	float lod = min(max(0.5f * log2f(d),0.0f),3.0f);

	int lod0 = floorf(lod);
	//float4 color0 = SampleTexture2DLOD(tex, uv, 0);
	float4 color0 = SampleTexture2DLOD(tex, uv, lod0);
	float4 color1 = SampleTexture2DLOD(tex, uv, min(lod0 + 1, 3));

	float factor = fracf(lod);
	result.x = color0.x * (1.0f - factor) + color1.x * factor;
	result.y = color0.y * (1.0f - factor) + color1.y * factor;
	result.z = color0.z * (1.0f - factor) + color1.z * factor;
	result.w = color0.w * (1.0f - factor) + color1.w * factor;
	//result = color0;
	return result;
}


#endif // !RASTER_FUNC
