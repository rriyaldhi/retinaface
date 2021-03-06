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

cuda_add_library(decodeplugin SHARED ${PROJECT_SOURCE_DIR}/decode.cu)
target_link_libraries(decodeplugin nvinfer cudart)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(inference ${PROJECT_SOURCE_DIR}/retinaface.cpp ${PROJECT_SOURCE_DIR}/inference.cpp)
target_link_libraries(inference nvinfer)
target_link_libraries(inference cudart)
target_link_libraries(inference decodeplugin)
target_link_libraries(inference ${OpenCV_LIBRARIES})

add_executable(inference_video ${PROJECT_SOURCE_DIR}/retinaface.cpp ${PROJECT_SOURCE_DIR}/inference_video.cpp)
target_link_libraries(inference_video nvinfer)
target_link_libraries(inference_video cudart)
target_link_libraries(inference_video decodeplugin)
target_link_libraries(inference_video ${OpenCV_LIBRARIES})

add_definitions(-O2 -pthread)

