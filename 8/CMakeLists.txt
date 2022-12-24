cmake_minimum_required(VERSION 3.21)
project(8)

file(GLOB src src/*.c*)
cuda_add_executable(${PROJECT_NAME} ${src})
