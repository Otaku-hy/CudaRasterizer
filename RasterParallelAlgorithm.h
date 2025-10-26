#pragma once

#include <algorithm>

#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include "RasterPipeHelper.h"
#include "ErrorCheck.h"

#define NOMINMAX

void InitCompactionEnvironment();

void  PrimitiveCompaction(int size, const Primitive* inputStream, Primitive* outputStream, unsigned int* sum, cudaStream_t stream);

void DestroyCompactionEnvironment();