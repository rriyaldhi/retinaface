cmake_minimum_required(VERSION 2.6)

project(retinaface)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

message("embed_platform off")
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(retina_r50 ${PROJECT_SOURCE_DIR}/retina_r50.cpp)
target_link_libraries(retina_r50 ${OpenCV_LIBRARIES})

add_definitions(-O2 -pthread)

