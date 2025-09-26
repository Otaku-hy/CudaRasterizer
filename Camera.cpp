#include "Camera.h"

Camera::Camera(glm::vec3 position, glm::vec3 lookPos, glm::vec3 up, float fovY, float aspectRatio, float zNear, float zFar)
	: mPosition(position), mFovY(fovY), mAspectRatio(aspectRatio), mZNear(zNear), mZFar(zFar)
{
	mLookDir = glm::normalize(lookPos - position);
	mRight = glm::normalize(glm::cross(mLookDir, up));
	mUp = glm::normalize(glm::cross(mRight, mLookDir));
}

glm::mat4 Camera::GetViewMatrix() const
{
	glm::mat4 view = glm::mat4(mRight.x, mRight.y, mRight.z, 0.0f,
		mUp.x, mUp.y, mUp.z, 0.0f,
		mLookDir.x, mLookDir.y, mLookDir.z, 0.0f,
		mPosition.x, mPosition.y, mPosition.z, 1.0f);
	return glm::inverse(view);
}

glm::mat4 Camera::GetProjectionMatrix() const
{
	float cosTheta = std::cos(mFovY * 0.5f);
	float sinTheta = std::sin(mFovY * 0.5f);
	float height = sinTheta / cosTheta;
	float width = height / mAspectRatio;

	float fRange = mZFar / (mZFar - mZNear);

	glm::mat4 proj = glm::mat4(width, 0, 0, 0,
		0, height, 0, 0,
		0, 0, fRange, 1.0f,
		0.0f, 0.0f, -fRange * mZNear, 0.0f);
	return proj;
}

void Camera::MoveCamera(MoveDirection dir, float deltaTime)
{
	switch (dir)
	{
	case MoveDirection::MOVE_FORWARD:
		mPosition += mLookDir * cameraMoveSpeed * deltaTime;
		break;
	case MoveDirection::MOVE_RIGHT:
		mPosition += mRight * cameraMoveSpeed * deltaTime;
		break;
	case MoveDirection::MOVE_UP:
		mPosition += mUp * cameraMoveSpeed * deltaTime;
		break;
	}
}

void Camera::RotateCamera(float xoffset, float yoffset)
{
	float anglePitch = yoffset * cameraRotateSpeed;
	float angleYaw = xoffset * cameraRotateSpeed;

	glm::mat4 rotX = glm::mat4(1.0f);
	rotX[1][1] = std::cos(anglePitch);
	rotX[1][2] = -std::sin(anglePitch);
	rotX[2][1] = std::sin(anglePitch);
	rotX[2][2] = std::cos(anglePitch);

	glm::mat4 rotY = glm::mat4(1.0f);
	rotY[0][0] = std::cos(angleYaw);
	rotY[0][2] = std::sin(angleYaw);
	rotY[2][0] = -std::sin(angleYaw);
	rotY[2][2] = std::cos(angleYaw);

	glm::mat4 rot = rotX * rotY;
	mLookDir = glm::normalize(glm::vec3(rot * glm::vec4(mLookDir, 0.0f)));
	mRight = glm::normalize(glm::vec3(rot * glm::vec4(mRight, 0.0f)));
	mUp = glm::normalize(glm::vec3(rot * glm::vec4(mUp, 0.0f)));
}