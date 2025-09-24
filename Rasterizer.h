#pragma once

#include <stdio.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "glm/glm/vec4.hpp"

#include "RasterPipeHelper.h"
#include "RasterMathHelper.h"
#include "ErrorCheck.h"

void InitializeCudaRasterizer(int width, int height);
void CleanupCudaRasterizer();

void Rasterize(cudaSurfaceObject_t outRenderTarget, float* depthBuffer,
	const VertexVSIn* vertexStream, const uint32_t* indexStream,
	int indexCount);

void RasterizerUpdateObjectsBuffer(int vertexCount, int primitiveCount);

