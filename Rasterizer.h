#pragma once

#include <stdio.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "glm/glm/vec4.hpp"

#include "RasterPipeHelper.h"
#include "RasterMathHelper.h"
#include "RasterParallelAlgorithm.h"
#include "RasterConst.h"
#include "ErrorCheck.h"

void InitializeCudaRasterizer(int width, int height);
void CleanupCudaRasterizer();

void Rasterize(unsigned char* outRenderTarget, float* depthBuffer,
	const VertexVSIn* vertexStream, const uint32_t* indexStream,
	int indexCount, int vertexCount, MatricesCBuffer* cb);

void RasterizerUpdateObjectsBuffer(int indexCountPerPrimitive, int vertexCount, int indexCount);
