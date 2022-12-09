cmake_minimum_required(VERSION 3.15)
project(yolov5_cpp)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(yolov5_cpp main.cpp)
target_link_libraries(yolov5_cpp ${OpenCV_LIBS})