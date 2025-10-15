#ifndef RASTER_MATH_HELPER
#define RASTER_MATH_HELPER

#include "glm/glm/glm.hpp"

#define UPPER_BOUND(x,n) ( ( (x) + ( (1<<(n)) -1) ) & ~( (1<<(n)) -1))

static constexpr int MAX_VERTEX_CLIP_COUNT = 16;

template<typename T>
struct is_glm_vec : std::false_type {};

template<> struct is_glm_vec<glm::vec2> : std::true_type {};
template<> struct is_glm_vec<glm::vec3> : std::true_type {};
template<> struct is_glm_vec<glm::vec4> : std::true_type {};

template<typename T>
	requires(is_glm_vec<T>::value)
struct AABB
{
	T min;
	T max;
};

template<typename T>
	requires(is_glm_vec<T>::value)
inline __host__ __device__ AABB<T> ComputeTriangleBoundingBox(const T v0, const T v1, const T v2)
{
	T min = glm::min(glm::min(v0, v1), v2);
	T max = glm::max(glm::max(v0, v1), v2);

	return { min, max };
}

template<typename T>
	requires(is_glm_vec<T>::value)
inline __host__ __device__ bool IsIntersected(const AABB<T> a, const AABB<T> b)
{
	return glm::all((a.min <= b.max) && (a.max >= b.min));
}

template<typename T>
	requires(is_glm_vec<T>::value)
inline __host__ __device__ AABB<T> Intersect(const AABB<T> a, const AABB<T> b)
{
	AABB<T> result;
	result.min = glm::max(a.min, b.min);
	result.max = glm::min(a.max, b.max);
	return result;
}

template<typename T>
	requires(is_glm_vec<T>::value)
inline __host__ __device__ AABB<T> Union(const AABB<T> a, const AABB<T> b)
{
	AABB<T> result;
	result.min = glm::min(a.min, b.min);
	result.max = glm::max(a.max, b.max);
	return result;
}


inline __host__ __device__ glm::vec4 Intersect(const glm::vec4 a, const glm::vec4 b)
{
	float rx = a.x > b.x ? a.x : b.x;
	float ry = a.y > b.y ? a.y : b.y;
	float rz = a.z < b.z ? a.z : b.z;
	float rw = a.w < b.w ? a.w : b.w;

	//float rx = glm::max(a.x, b.x);
	//float ry = glm::max(a.y, b.y);
	//float rz = glm::min(a.z, b.z);
	//float rw = glm::min(a.w, b.w);
	return glm::vec4(rx, ry, rz, rw);
}

inline __host__ __device__ glm::vec3 ComputeBarycentric2D(const glm::vec2 p, const glm::vec2 v0, const glm::vec2 v1, const glm::vec2 v2)
{
	float s = (v1.y - v2.y) * v0.x + (v2.x - v1.x) * v0.y + v1.x * v2.y - v2.x * v1.y;
	float invS = 1.0f / s;

	float s0 = (v1.y - v2.y) * p.x + (v2.x - v1.x) * p.y + v1.x * v2.y - v2.x * v1.y;
	float s1 = (v2.y - v0.y) * p.x + (v0.x - v2.x) * p.y + v2.x * v0.y - v0.x * v2.y;

	return { s0 * invS, s1 * invS, 1 - (s0 + s1) * invS };
}

inline __device__ bool InsideViewFrustumCS(const glm::vec4& positionCS)
{
	return positionCS.x >= -positionCS.w && positionCS.x <= positionCS.w &&
		positionCS.y >= -positionCS.w && positionCS.y <= positionCS.w &&
		positionCS.z >= 0.0f && positionCS.z <= positionCS.w;
}

inline __device__ VertexVSOut InterpolatingClippingPos(float t, VertexVSOut  vStart, VertexVSOut vEnd)
{
	VertexVSOut vResult;
	vResult.sv_position = glm::mix(vStart.sv_position, vEnd.sv_position, t); //lerp
	return vResult;
}

inline __device__ int ClippingWithPlane(const glm::vec4 faceEquation, const int numVertices, const VertexVSOut* inVertices, VertexVSOut* outVertices)
{
	glm::vec4 prevPos = inVertices[numVertices - 1].sv_position;
	VertexVSOut prevVertex = inVertices[numVertices - 1];
	int outVertexCount = 0;

	for (int i = 0; i < numVertices; i++)
	{
		glm::vec4 currPos = inVertices[i].sv_position;
		float disCurrPos = glm::dot(faceEquation, currPos);
		float disPrevPos = glm::dot(faceEquation, prevPos);

		float t = disPrevPos / (disPrevPos - disCurrPos);
		if ((disCurrPos > 0.0f) != (disPrevPos > 0.0f))
			outVertices[outVertexCount++] = InterpolatingClippingPos(t, prevVertex, inVertices[i]);
		if (disCurrPos >= 0.0f) outVertices[outVertexCount++] = inVertices[i];

		prevPos = currPos;
		prevVertex = inVertices[i];
	}

	return outVertexCount;
}

inline __device__ __host__ unsigned NormToUnsigned_24Bit(float val)
{
	const float max24Bit = 16777215.0f; // 2^24 -1
	return static_cast<unsigned>(val * max24Bit);
}

inline __device__ __host__ float UnsignedToNorm_24Bit(unsigned val)
{
	const float invMax24Bit = 1.0f / 16777215.0f; // 1/(2^24 -1)
	return static_cast<float>(val) * invMax24Bit;
}

#endif // !RASTER_MATH_HELPER
