#pragma once

#include <algorithm>

#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include "RasterPipeHelper.h"
#include "ErrorCheck.h"

#define NOMINMAX

void InitCompactionEnvironment();

int  PrimitiveCompaction(int size, const Primitive* inputStream, Primitive* outputStream);

void DestroyCompactionEnvironment();