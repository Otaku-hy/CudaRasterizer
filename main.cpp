#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <windows.h> // For MessageBoxA

#define GLEW_STATIC
#include "GL/glew.h"
#include "GLFW/glfw3.h"
#include "glm/glm/glm.hpp"
#include "glm/glm/gtc/matrix_transform.hpp"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "ErrorCheck.h"
#include "ObjLoader.h"
#include "glUtility.h"
#include "Rasterizer.h" 
#include "Camera.h"

#include "ParallelAlgorithm.h"

namespace
{
	// Global variables from your project
	int width = 1920;
	int height = 1080;

	VertexVSIn* gpInVertexStream = nullptr;
	uint32_t* gpIndexStream = nullptr;
	unsigned* gpCudaDepthStencil = nullptr;
	GLuint gRTBuffer;
	GLuint gRTTexture;
	uint32_t gIndexCount = 0;
	uint32_t gVertexCount = 0;

	// OpenGL/GLFW specific globals
	GLFWwindow* gpWindow = nullptr;
	GLuint gShaderProgram;
	GLuint gRenderQuadVAO;

	float deltaTime = 0.0f;
	float lastFrameTime = 0.0f;

	bool firstRotatePos = true;
	bool isMousePressed = false;

	float xPos;
	float yPos;

	std::unique_ptr<Camera> gpCamera;
}

// Function Prototypes
void Initialize();
void Cleanup();
void RenderLoop();
void Update();
void Rendering();
void LoadAssets();

void ProcessInput(GLFWwindow* window);

void MousePositionCallback(GLFWwindow* window, double xpos, double ypos);
void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods);

// Simple shaders to draw a textured quad
const char* vertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec2 aPos;
    layout (location = 1) in vec2 aTexCoords;
    out vec2 TexCoords;
    void main() {
        gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);
        TexCoords = aTexCoords;
    }
)";
const char* fragmentShaderSource = R"(
    #version 330 core
    out vec4 FragColor;
    in vec2 TexCoords;
    uniform sampler2D screenTexture;
    void main() {
        FragColor = texture(screenTexture, TexCoords);
    }
)";

int main()
{
	try
	{
		Initialize();
		InitializeCudaRasterizer(width, height);
		LoadAssets();
		RenderLoop();
	}
	catch (const std::exception& e)
	{
		OutputDebugStringA(e.what());
		Cleanup();
		CleanupCudaRasterizer();
		return -1;
	}

	Cleanup();
	CleanupCudaRasterizer();
	return 0;
}

void Initialize()
{
	if (!glfwInit())
	{
		throw std::runtime_error("Failed to initialize GLFW.");
	}
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	gpWindow = glfwCreateWindow(width, height, "CUDA Rasterizer with OpenGL", NULL, NULL);
	if (!gpWindow)
	{
		glfwTerminate();
		throw std::runtime_error("Failed to create GLFW window.");
	}
	glfwMakeContextCurrent(gpWindow);
	glfwSwapInterval(1);

	glfwSetInputMode(gpWindow, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
	glfwSetCursorPosCallback(gpWindow, MousePositionCallback);
	glfwSetMouseButtonCallback(gpWindow, MouseButtonCallback);

	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK)
	{
		throw std::runtime_error("Failed to initialize GLEW.");
	}

	CUDA_CHECK(cudaSetDevice(0));

	CompileShaders(vertexShaderSource, fragmentShaderSource, &gShaderProgram);

	float quadVertices[] = {
		-1.0f,  1.0f,  0.0f, 1.0f, -1.0f, -1.0f,  0.0f, 0.0f, 1.0f, -1.0f,  1.0f, 0.0f,
		-1.0f,  1.0f,  0.0f, 1.0f,  1.0f, -1.0f,  1.0f, 0.0f, 1.0f,  1.0f,  1.0f, 1.0f
	};
	GLuint quad_vbo;
	glGenVertexArrays(1, &gRenderQuadVAO);
	glGenBuffers(1, &quad_vbo);
	glBindVertexArray(gRenderQuadVAO);
	glBindBuffer(GL_ARRAY_BUFFER, quad_vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));

	const unsigned int buffer_size = width * height * 4 * sizeof(unsigned char);
	glGenBuffers(1, &gRTBuffer);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, gRTBuffer);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, buffer_size, NULL, GL_DYNAMIC_COPY);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	CUDA_CHECK(cudaGLRegisterBufferObject(gRTBuffer));

	glGenTextures(1, &gRTTexture);
	glBindTexture(GL_TEXTURE_2D, gRTTexture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glBindTexture(GL_TEXTURE_2D, 0);

	glViewport(0, 0, width, height);
}

void LoadAssets()
{
	objl::Loader objLoader;
	objLoader.LoadFile("./objs/cow.obj");
	//std::vector<VertexVSIn> rasterVertices = {
	//	{ glm::vec4{-0.1f, -0.1f, 0.5f, 1.0f} },
	//	{ glm::vec4{0.0f, 0.1f, 0.5f, 1.0f} },
	//	{ glm::vec4{0.1f, -0.1f, 0.5f, 1.0f} }
	//}; ;
	//std::vector<unsigned int> rasterIndices = { 0,2,1 };
	std::vector<VertexVSIn> rasterVertices;
	std::vector<unsigned int> rasterIndices;
	LoadObjToRasterStruct(objLoader.LoadedVertices, rasterVertices);
	rasterIndices = objLoader.LoadedIndices;

	gVertexCount = static_cast<uint32_t>(rasterVertices.size());
	gIndexCount = static_cast<uint32_t>(rasterIndices.size());

	gpCamera = std::make_unique<Camera>(glm::vec3(0.0f, 0.0f, 0.37f), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0), glm::radians(60.0f), static_cast<float>(width) / height);

	CUDA_CHECK(cudaMalloc((void**)&gpCudaDepthStencil, sizeof(unsigned) * width * height));

	VertexVSIn vertices[] =
	{
		{ glm::vec4{-0.1f, -0.1f, 0.5f, 1.0f} },
		{ glm::vec4{0.0f, 0.1f, 0.5f, 1.0f} },
		{ glm::vec4{0.1f, -0.1f, 0.5f, 1.0f} }
	};
	uint32_t indices[] = { 0,2,1 };

	RasterizerUpdateObjectsBuffer(3, gVertexCount, gIndexCount);

	CUDA_CHECK(cudaMalloc((void**)&gpInVertexStream, sizeof(VertexVSIn) * rasterVertices.size()));
	CUDA_CHECK(cudaMemcpy((void*)gpInVertexStream, rasterVertices.data(), sizeof(VertexVSIn) * rasterVertices.size(), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMalloc((void**)&gpIndexStream, sizeof(unsigned int) * rasterIndices.size()));
	CUDA_CHECK(cudaMemcpy((void*)gpIndexStream, rasterIndices.data(), sizeof(unsigned int) * rasterIndices.size(), cudaMemcpyHostToDevice));
}


void RenderLoop()
{
	//while (!glfwWindowShouldClose(gpWindow))
	while (!glfwWindowShouldClose(gpWindow))
	{
		float currentFrameTime = static_cast<float>(glfwGetTime());
		deltaTime = currentFrameTime - lastFrameTime;
		lastFrameTime = currentFrameTime;

		ProcessInput(gpWindow);

		Update();
		Rendering();

		glfwSwapBuffers(gpWindow);
		glfwPollEvents();

		//break;
	}
}

void Update()
{
	// update scene & logic here
}

void Rendering()
{
	//begin pass
	unsigned char* cudaMappedRT = nullptr;
	CUDA_CHECK(cudaGLMapBufferObject((void**)&cudaMappedRT, gRTBuffer));

	MatricesCBuffer cb;
	glm::mat4 view = gpCamera->GetViewMatrix();
	glm::mat4 proj = gpCamera->GetProjectionMatrix();
	cb.mvp = proj * view; //column major

	//draw call with cuda rasterizer
	Rasterize(cudaMappedRT, gpCudaDepthStencil,
		gpInVertexStream, gpIndexStream, gIndexCount, gVertexCount, &cb);
	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_CHECK(cudaGLUnmapBufferObject(gRTBuffer));

	//end pass 
	//copy to texture and render textured quad
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, gRTBuffer);
	glBindTexture(GL_TEXTURE_2D, gRTTexture);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	glUseProgram(gShaderProgram);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, gRTTexture);
	glBindVertexArray(gRenderQuadVAO);
	glDrawArrays(GL_TRIANGLES, 0, 6);

	glBindVertexArray(0);
	glBindTexture(GL_TEXTURE_2D, 0);
}

void Cleanup()
{
	if (gpWindow)
	{
		glfwMakeContextCurrent(gpWindow);
	}

	if (gRTBuffer)
	{
		CUDA_CHECK(cudaGLUnregisterBufferObject(gRTBuffer));
	}

	// Free your CUDA buffers
	cudaFree(gpCudaDepthStencil);
	cudaFree(gpInVertexStream);
	cudaFree(gpIndexStream);

	glDeleteProgram(gShaderProgram);
	glDeleteTextures(1, &gRTTexture);
	glDeleteBuffers(1, &gRTBuffer);
	glDeleteVertexArrays(1, &gRenderQuadVAO);

	if (gpWindow)
	{
		glfwDestroyWindow(gpWindow);
	}
	glfwTerminate();
}

void ProcessInput(GLFWwindow* window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);

	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		gpCamera->MoveCamera(MoveDirection::MOVE_FORWARD, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		gpCamera->MoveCamera(MoveDirection::MOVE_FORWARD, -deltaTime);
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		gpCamera->MoveCamera(MoveDirection::MOVE_RIGHT, -deltaTime);
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		gpCamera->MoveCamera(MoveDirection::MOVE_RIGHT, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
		gpCamera->MoveCamera(MoveDirection::MOVE_UP, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
		gpCamera->MoveCamera(MoveDirection::MOVE_UP, -deltaTime);
}

void MousePositionCallback(GLFWwindow* window, double xpos, double ypos)
{
	if (!isMousePressed) return;

	if (firstRotatePos)
	{
		firstRotatePos = false;
		xPos = static_cast<float>(xpos);
		yPos = static_cast<float>(ypos);
	}
	else
	{
		float xoffset = static_cast<float>(xpos) - xPos;
		float yoffset =  static_cast<float>(ypos)  - yPos;

		gpCamera->RotateCamera(xoffset, yoffset);

		xPos = static_cast<float>(xpos);
		yPos = static_cast<float>(ypos);
	}
}

void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
	{
		isMousePressed = true;
		firstRotatePos = true;
	}
	else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE)
	{
		isMousePressed = false;
	}
}
