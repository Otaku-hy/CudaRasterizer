#pragma once

#include <stdexcept>
#include <sstream>
#include <iostream>

#include <windows.h>
#include <comdef.h>
#include <cuda_runtime.h>

inline void ThrowIfFailed(HRESULT hr, const char* funcCall, const char* file, int line)
{
	if (FAILED(hr))
	{
		_com_error err(hr);

		std::ostringstream oss;
		oss << "Error at " << file << ":" << line << " - " << funcCall << " failed with HRESULT: " << err.ErrorMessage();
		//fprintf(stderr, "%s\n", oss.str().c_str());
		throw std::runtime_error(oss.str());
	}
}

inline void CheckCudaError(cudaError_t err, const char* funcCall, const char* file, int line)
{
	if (err != cudaSuccess)
	{
		std::ostringstream oss;
		oss << "CUDA Error at " << file << ":" << line << " - " << funcCall << " failed with:  " << cudaGetErrorString(err);
		//fprintf(stderr, "%s\n", oss.str().c_str());
		//exit(EXIT_FAILURE);
		OutputDebugStringA(oss.str().c_str());
		throw std::runtime_error(oss.str());
	}
}

#if defined(_DEBUG)
#define DX_CHECK(x) { ThrowIfFailed((x), #x, __FILE__, __LINE__); }
#else
#define DX_CHECK(x) (x)
#endif

#if defined(_DEBUG)
#define CUDA_CHECK(x) { CheckCudaError((x), #x, __FILE__, __LINE__); }
#else
#define CUDA_CHECK(x) (x)
#endif