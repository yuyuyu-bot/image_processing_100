cmake_minimum_required(VERSION 3.21)
project(2-rgb_to_gray)

file(GLOB src src/*.c*)
cuda_add_executable(${PROJECT_NAME} ${src})
