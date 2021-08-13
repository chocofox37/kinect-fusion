cmake_minimum_required(VERSION 3.0.0)
project(kinect-fusion LANGUAGES CXX CUDA)

find_package(Eigen3 REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_libraries(${OpenCV_LIBS})

file(GLOB SOURCE_FILES ${PROJECT_SOURCE_DIR}/*.cu)
add_executable(${PROJECT_NAME} ${SOURCE_FILES})
