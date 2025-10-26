#pragma once

#include "Rasterizer.h"

void InitializeCudaGraph();

void CleanupCudaGraph();

void BuildPipeline(GLuint rtBuffer, unsigned* depthBuffer,
	const VertexVSIn* vertexStream, const uint32_t* indexStream,
	int indexCount, int vertexCount, MatricesCBuffer* cb, Texture2D tex);

void RasterizeWithGraph(GLuint rtBuffer, unsigned* depthBuffer,
	const VertexVSIn* vertexStream, const uint32_t* indexStream,
	int indexCount, int vertexCount, MatricesCBuffer* cb, Texture2D tex);