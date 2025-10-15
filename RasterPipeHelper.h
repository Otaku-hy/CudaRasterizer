#ifndef RASTER_PIPE_HELPER_H
#define RASTER_PIPE_HELPER_H

#include "glm/glm/glm.hpp"

struct VertexVSIn
{
	glm::vec4 position;
};

struct VertexVSOut
{
	glm::vec4 sv_position;
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
	//glm::vec3 normalEq;
};

struct FragmentPSin
{
	unsigned int primitiveID;
	unsigned int mask;
	glm::vec4 sv_position; //xy: pixel center position, z: depth

	// other attributes interpolation
	glm::vec4 color;
};

struct FragmentPSOut
{
	unsigned int primitiveID;
	unsigned int mask;
	unsigned int x;
	unsigned int y;
	unsigned int depth;

	glm::vec4 sv_position;
	glm::vec4 color;
};

#endif // !RASTER_PIPE_HELPER_H

