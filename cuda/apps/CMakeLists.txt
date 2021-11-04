cmake_minimum_required(VERSION 3.20)
project(mlib-cuda-apps LANGUAGES CUDA CXX)



add_executable(mbm mbm_main.cpp )
target_link_libraries(mbm mlib-cuda)


