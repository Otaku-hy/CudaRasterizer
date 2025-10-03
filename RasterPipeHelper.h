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

struct FragmentPSin
{
	glm::vec4 color;
};

#endif // !RASTER_PIPE_HELPER_H

