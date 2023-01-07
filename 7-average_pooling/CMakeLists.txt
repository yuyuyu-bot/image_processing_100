cmake_minimum_required(VERSION 3.21)
project(7-average_pooling)

file(GLOB src src/*.c*)
cuda_add_executable(${PROJECT_NAME} ${src})
