#pragma once

#include "glm//glm//glm.hpp"

enum class MoveDirection: uint8_t
{
	MOVE_FORWARD,
	MOVE_RIGHT,
	MOVE_UP,
};

// perspective camera with left handed coordinate system (front, right, up)
class Camera
{
public:
	Camera() = default;
	Camera(glm::vec3 position, glm::vec3 lookPos, glm::vec3 up, float fovY, float aspectRatio, float zNear = 0.1f, float zFar = 100.0f);

	glm::mat4 GetViewMatrix() const;
	glm::mat4 GetProjectionMatrix() const;

	void MoveCamera(MoveDirection dir, float deltaTime);
	void RotateCamera(float xoffset, float yoffset);

private:     
	glm::vec3 mPosition;
	glm::vec3  mLookDir;
	glm::vec3 mUp;
	glm::vec3 mRight;

	float mFovY;
	float mAspectRatio;
	float mZNear;
	float mZFar;

	float cameraMoveSpeed = 1.0f;
	float cameraRotateSpeed = 0.01f;
};

