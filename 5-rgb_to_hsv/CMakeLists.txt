cmake_minimum_required(VERSION 3.21)
project(5-rgb_to_hsv)

file(GLOB src src/*.c*)
cuda_add_executable(${PROJECT_NAME} ${src})
