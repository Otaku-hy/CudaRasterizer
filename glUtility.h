#pragma once

#include <stdexcept>

#include "GL/glew.h"
#include "GLFW/glfw3.h"

#include "ObjLoader.h"
#include "RasterPipeHelper.h"

inline void CompileShaders(const char* vertexShaderSource, const char* fragmentShaderSource, GLuint* shaderProgram)
{
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        throw std::runtime_error(std::string("Vertex shader compilation failed:\n") + infoLog);
    }

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        throw std::runtime_error(std::string("Fragment shader compilation failed:\n") + infoLog);
    }

    *shaderProgram = glCreateProgram();
    glAttachShader(*shaderProgram, vertexShader);
    glAttachShader(*shaderProgram, fragmentShader);
    glLinkProgram(*shaderProgram);

    glGetProgramiv(*shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(*shaderProgram, 512, NULL, infoLog);
        throw std::runtime_error(std::string("Shader program linking failed:\n") + infoLog);
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    glUseProgram(*shaderProgram);
    glUniform1i(glGetUniformLocation(*shaderProgram, "screenTexture"), 0);
}

inline void LoadObjToRasterStruct(const std::vector<objl::Vertex>& inVertexStream, std::vector<VertexVSIn>& outVertexStream)
{
    outVertexStream.resize(inVertexStream.size());
    for (size_t i = 0; i < inVertexStream.size(); ++i) {
        outVertexStream[i].position = glm::vec4(inVertexStream[i].Position.X, inVertexStream[i].Position.Y, inVertexStream[i].Position.Z, 1.0f);
	}
}

