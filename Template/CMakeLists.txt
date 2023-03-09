cmake_minimum_required(VERSION 3.16)
project(AI_Template)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-fPIC -w -pthread")

# Engine
add_library(engine SHARED main.cpp)

# TensorRT
set(CUDA_include "/usr/local/cuda-11.3/include")
set(TensorRT_sample_include "/home/kangshuai/Desktop/industry_detec/cpp/TensorRT-8.0.1.6.Linux.x86_64-gnu.cuda-11.3.cudnn8.2/samples/common")
set(TensorRT_include "/home/kangshuai/Desktop/industry_detec/cpp/TensorRT-8.0.1.6.Linux.x86_64-gnu.cuda-11.3.cudnn8.2/include")
include_directories(${CUDA_include} ${TensorRT_common} ${TensorRT_include})

set(LIB_CUDA "/usr/local/cuda-11.3/lib64")
set(LIB_TENSORRT "/home/kangshuai/Desktop/industry_detec/cpp/TensorRT-8.0.1.6.Linux.x86_64-gnu.cuda-11.3.cudnn8.2/lib")

find_library(libnvinfer nvinfer ${LIB_TENSORRT})
message(STATUS "Found nvinfer: ${libnvinfer}")

find_library(libnvinfer_plugin nvinfer_plugin ${LIB_TENSORRT})
message(STATUS "Found nvinfer_plugin: ${libnvinfer_plugin}")

find_library(libcudart libcudart.so.11.0 ${LIB_CUDA})
message(STATUS "Found cudart: ${libcudart}")

add_library(trt OBJECT ${CMAKE_SOURCE_DIR}/src/trt.cpp)
target_link_libraries(trt ${libnvinfer} ${libnvinfer_plugin} ${libcudart})

target_link_libraries(engine trt)

# OnnxRT
set(onnx_include "/home/kangshuai/Desktop/industry_detec/cpp/onnxruntime-linux-x64-1.8.1/include")
include_directories(${onnx_include})

set(LIB_ONNX "/home/kangshuai/Desktop/industry_detec/cpp/onnxruntime-linux-x64-1.8.1/lib")

find_library(libonnxruntime onnxruntime ${LIB_ONNX})
message(STATUS "Found onnxruntime: ${libonnxruntime}")

add_library(onnxrt OBJECT ${CMAKE_SOURCE_DIR}/src/onnxrt_c.cpp)
target_link_libraries(onnxrt ${libonnxruntime})

target_link_libraries(engine onnxrt)

set(OpenCV_DIR "/home/kangshuai/Desktop/industry_detec/cpp/opencv4.5.3/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)

set(OpenCV_include "/home/kangshuai/Desktop/industry_detec/cpp/opencv4.5.3/include/opencv4")
include_directories(${OpenCV_include})

add_executable(main main.cpp)
target_link_libraries(main engine ${OpenCV_LIBS})













