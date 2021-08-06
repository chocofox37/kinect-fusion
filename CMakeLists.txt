cmake_minimum_required(VERSION 3.0.0)
project(kinect-fusion)

find_package(Eigen3 REQUIRED)
list(APPEND PROJECT_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
list(APPEND PROJECT_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})
list(APPEND PROJECT_LIBS ${OpenCV_LIBS})

include_directories(${PROJECT_INCLUDE_DIRS})
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} ${PROJECT_LIBS})
