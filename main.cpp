#include <memory>
#include <vector>
#include <type_traits>
#include <array>

#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <D3Dcompiler.h>
#include <DirectXMath.h>
#include <wrl.h>
#include <wtypesbase.h>

#include "ErrorCheck.h"
#include "Rasterizer.h"

template<typename T>
	requires(!std::is_lvalue_reference<T>::value)
T* get_rvalue_ptr(T&& v)
{
	return &v;
}

namespace
{
	using Microsoft::WRL::ComPtr;

	int width = 1280;
	int height = 720;

	HWND hwnd = nullptr;

	std::unique_ptr<ID3D12Device> gpDevice;
	std::unique_ptr<ID3D12CommandQueue> gpCommandQueue;
	std::unique_ptr<ID3D12GraphicsCommandList> gpCommandList;
	std::unique_ptr<ID3D12CommandAllocator> gpCommandAllocator;

	std::array<std::unique_ptr<ID3D12Resource>, 2> gpRenderTargets;
	std::unique_ptr<IDXGISwapChain3> gpSwapChain;
	uint32_t gCurrentBackBuffer = 0;

	uint32_t gFenceValue = 0;
	std::unique_ptr<ID3D12Fence> gpFence;

	cudaExternalMemory_t gCudaExternalMemory = nullptr;
	cudaMipmappedArray_t gCudaMipmappedArray = nullptr;
	cudaArray_t gCudaTexArray = nullptr;      //the real cuda render target!!!!!!!
	std::unique_ptr<ID3D12Resource> gpCudaRenderTarget;

	VertexVSIn* gpInVertexStream = nullptr;
	uint32_t* gpIndexStream = nullptr;
	float* gpCudaDepthStencil = nullptr;
}

void InitializeD3D12();
void DestoryRasterizer();
void RenderLoop();
void Update();		//use to update scene & logic
void Rendering();			//main rendering function
void BeginFrame();		 //prepare rendering -> update const buffers
void RenderPass();		//bind resources to cuda rasterizer & rasterize
void EndFrame();			//populate cmdlist & execute & present & synchronize
void Synchronize();

void LoadAssets();

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int nCmdShow)
{
	WNDCLASSEX wc = { sizeof(WNDCLASSEX), CS_CLASSDC, WindowProc, 0L, 0L, GetModuleHandle(NULL), NULL, NULL, NULL, NULL, "CUDA Rasterizer Window", NULL };
	RegisterClassEx(&wc);

	HANDLE hWnd = CreateWindow("CUDA Rasterizer Window", "CUDA Rasterizer", WS_OVERLAPPEDWINDOW, 100, 100, width, height, NULL, NULL, wc.hInstance, NULL);

	InitializeD3D12();
	InitializeCudaRasterizer(width, height);
	LoadAssets();

	ShowWindow(hwnd, nCmdShow);
	UpdateWindow(hwnd);

	RenderLoop();

	CleanupCudaRasterizer();
	DestoryRasterizer();

	return 0;
}

void RenderLoop()
{
	MSG msg{};
	while (msg.message != WM_QUIT)
	{
		if (PeekMessage(&msg, hwnd, 0, 0, PM_REMOVE))
		{
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
		else
		{
			Update();
			Rendering();
		}
	}

}

void Update()
{
	//update scene & logic
}

void Rendering()
{
	BeginFrame();
	RenderPass();
	EndFrame();
}

void BeginFrame()
{
	gpCommandAllocator->Reset();
	gpCommandList->Reset(gpCommandAllocator.get(), nullptr);

}

void EndFrame()
{
	gpCommandList->Close();

	ID3D12CommandList* ppCommandLists[] = { gpCommandList.get() };
	gpCommandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

	DX_CHECK(gpSwapChain->Present(1, 0));
	Synchronize();
}

void RenderPass()
{
	//begin pass
	cudaResourceDesc resDesc{};
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = gCudaTexArray;
	cudaSurfaceObject_t renderTarget;
	CUDA_CHECK(cudaCreateSurfaceObject(&renderTarget, &resDesc));

	//rasterize
	Rasterize(renderTarget, gpCudaDepthStencil,
		gpInVertexStream, gpIndexStream, 3);
	CUDA_CHECK(cudaDeviceSynchronize());

	//end pass
	gCurrentBackBuffer = gpSwapChain->GetCurrentBackBufferIndex();

	D3D12_RESOURCE_BARRIER barrier1{};
	barrier1.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	barrier1.Transition.pResource = gpCudaRenderTarget.get();
	barrier1.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
	barrier1.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
	barrier1.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
	gpCommandList->ResourceBarrier(1, &barrier1);

	D3D12_RESOURCE_BARRIER barrier2{};
	barrier2.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	barrier2.Transition.pResource = gpRenderTargets[gCurrentBackBuffer].get();
	barrier2.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
	barrier2.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
	barrier2.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
	gpCommandList->ResourceBarrier(1, &barrier2);

	gpCommandList->CopyResource(gpRenderTargets[gCurrentBackBuffer].get(), gpCudaRenderTarget.get());

	D3D12_RESOURCE_BARRIER barrier3{};
	barrier3.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	barrier3.Transition.pResource = gpCudaRenderTarget.get();
	barrier3.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
	barrier3.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
	barrier3.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
	gpCommandList->ResourceBarrier(1, &barrier3);

	D3D12_RESOURCE_BARRIER barrier4{};
	barrier4.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	barrier4.Transition.pResource = gpRenderTargets[gCurrentBackBuffer].get();
	barrier4.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
	barrier4.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
	barrier4.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
	gpCommandList->ResourceBarrier(1, &barrier4);

	CUDA_CHECK(cudaDestroySurfaceObject(renderTarget));
}

void Synchronize()
{
	const uint64_t fence = gFenceValue;
	DX_CHECK(gpCommandQueue->Signal(gpFence.get(), fence));
	gFenceValue++;
	if (gpFence->GetCompletedValue() < fence)
	{
		HANDLE eventHandle = CreateEvent(nullptr, FALSE, FALSE, nullptr);
		DX_CHECK(gpFence->SetEventOnCompletion(fence, eventHandle));
		WaitForSingleObject(eventHandle, INFINITE);
		CloseHandle(eventHandle);
	}
}

void InitializeD3D12()
{
#if defined(_DEBUG)
	ComPtr<ID3D12Debug> debugController;
	if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController))))
		debugController->EnableDebugLayer();
#endif
	ComPtr<IDXGIFactory7> pFactory;
	DX_CHECK(CreateDXGIFactory2(0, IID_PPV_ARGS(&pFactory)));

	{
		ID3D12Device* pDevice = gpDevice.get();
		DX_CHECK(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&pDevice)));
		gpDevice.reset(pDevice);
	}

	D3D12_COMMAND_QUEUE_DESC cqDesc = {};
	cqDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
	cqDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
	ID3D12CommandQueue* pCommandQueue = gpCommandQueue.get();
	DX_CHECK(gpDevice->CreateCommandQueue(&cqDesc, IID_PPV_ARGS(&pCommandQueue)));

	DX_CHECK(gpDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(get_rvalue_ptr(gpCommandAllocator.get()))));
	DX_CHECK(gpDevice->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, gpCommandAllocator.get(), nullptr, IID_PPV_ARGS(get_rvalue_ptr(gpCommandList.get()))));

	DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {};
	swapChainDesc.BufferCount = 2;
	swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	swapChainDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
	swapChainDesc.SampleDesc.Count = 1;
	swapChainDesc.Width = width;
	swapChainDesc.Height = height;

	IDXGISwapChain1* pSwapChain;
	DX_CHECK(pFactory->CreateSwapChainForHwnd(gpCommandQueue.get(), hwnd, &swapChainDesc, nullptr, nullptr, &pSwapChain));
	gpSwapChain.reset(static_cast<IDXGISwapChain3*>(pSwapChain));
	gCurrentBackBuffer = gpSwapChain->GetCurrentBackBufferIndex();

	for (size_t i = 0; i < swapChainDesc.BufferCount; i++)
	{
		ID3D12Resource* pBackBuffer;
		DX_CHECK(gpSwapChain->GetBuffer(i, IID_PPV_ARGS(&pBackBuffer)));
		gpRenderTargets[i].reset(pBackBuffer);
	}

	D3D12_HEAP_PROPERTIES heapProp = {};
	heapProp.Type = D3D12_HEAP_TYPE_DEFAULT;

	D3D12_RESOURCE_DESC resDesc = {};
	resDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
	resDesc.Width = width;
	resDesc.Height = height;
	resDesc.MipLevels = 1;
	resDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	resDesc.SampleDesc.Count = 1;
	resDesc.SampleDesc.Quality = 0;
	resDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
	resDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER;

	DX_CHECK(gpDevice->CreateCommittedResource(&heapProp,
		D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER,
		&resDesc,
		D3D12_RESOURCE_STATE_COMMON,
		nullptr,
		IID_PPV_ARGS(get_rvalue_ptr(gpCudaRenderTarget.get()))));

	DX_CHECK(gpDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(get_rvalue_ptr(gpFence.get()))));
	gFenceValue++;

	HANDLE sharedHandle = nullptr;
	DX_CHECK(gpDevice->CreateSharedHandle(
		gpCudaRenderTarget.get(), nullptr, GENERIC_ALL, nullptr, &sharedHandle));

	cudaExternalMemoryHandleDesc externalMemoryHandleDesc{};
	externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeD3D12Resource;
	externalMemoryHandleDesc.handle.win32.handle = sharedHandle;
	externalMemoryHandleDesc.size = 4 * width * height;  /*4 8bits component*/
	externalMemoryHandleDesc.flags = cudaExternalMemoryDedicated;
	CUDA_CHECK(cudaImportExternalMemory(&gCudaExternalMemory, &externalMemoryHandleDesc));

	cudaExternalMemoryMipmappedArrayDesc mipmappedArrayDesc{};
	mipmappedArrayDesc.extent = make_cudaExtent(width, height, 0);
	mipmappedArrayDesc.numLevels = 1;
	mipmappedArrayDesc.formatDesc.f = cudaChannelFormatKindUnsigned;
	mipmappedArrayDesc.formatDesc.x = 8;
	mipmappedArrayDesc.formatDesc.y = 8;
	mipmappedArrayDesc.formatDesc.z = 8;
	mipmappedArrayDesc.formatDesc.w = 8;
	CUDA_CHECK(cudaExternalMemoryGetMappedMipmappedArray(&gCudaMipmappedArray, gCudaExternalMemory, &mipmappedArrayDesc));

	CUDA_CHECK(cudaGetMipmappedArrayLevel(&gCudaTexArray, gCudaMipmappedArray, 0));
	CloseHandle(sharedHandle);
}

void DestoryRasterizer()
{
	Synchronize();

	cudaFreeMipmappedArray(gCudaMipmappedArray);
	cudaDestroyExternalMemory(gCudaExternalMemory);

	cudaFree(gpCudaDepthStencil);
	cudaFree(gpInVertexStream);
	cudaFree(gpIndexStream);
}

void LoadAssets()
{
	///depth stencil buffer
	CUDA_CHECK(cudaMalloc((void**)&gpCudaDepthStencil, sizeof(float) * width * height));

	VertexVSIn vertices[] =
	{
		{ glm::vec4{-0.5f,-0.5f,0.0f,0.5f}},
		{ glm::vec4(0.0f,0.5f,0.0f,0.5f)},
		{ glm::vec4(0.5f,-0.5f,0.0f,0.5f)}
	};
	uint32_t indices[] = { 0,1,2 };

	RasterizerUpdateObjectsBuffer(3, 3);

	//// vertex & index buffer
	CUDA_CHECK(cudaMalloc((void**)&gpInVertexStream, sizeof(vertices)));
	CUDA_CHECK(cudaMemcpy((void*)gpInVertexStream, vertices, sizeof(vertices), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMalloc((void**)&gpIndexStream, sizeof(indices)));
	CUDA_CHECK(cudaMemcpy((void*)gpIndexStream, indices, sizeof(indices), cudaMemcpyHostToDevice));
}

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	switch (uMsg)
	{
	case WM_DESTROY:
		PostQuitMessage(0);
		return 0;
	case WM_KEYDOWN:
		if (wParam == VK_ESCAPE)
			DestroyWindow(hwnd);
		return 0;
	}
	return DefWindowProc(hwnd, uMsg, wParam, lParam);
}