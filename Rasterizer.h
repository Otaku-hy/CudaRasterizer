#pragma once

#define NOMINMAX

#include <stdio.h>
#include <algorithm>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "glm/glm/vec4.hpp"

#include "RasterPipeHelper.h"
#include "RasterMathHelper.h"
#include "RasterParallelAlgorithm.h"
#include "RasterConstant.h"
#include "ErrorCheck.h"

void InitializeCudaRasterizer(int width, int height);
void CleanupCudaRasterizer();

void Rasterize(unsigned char* outRenderTarget, unsigned* depthBuffer,
	const VertexVSIn* vertexStream, const uint32_t* indexStream,
	int indexCount, int vertexCount, MatricesCBuffer* cb);

void RasterizerUpdateObjectsBuffer(int indexCountPerPrimitive, int vertexCount, int indexCount);
