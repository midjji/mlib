cmake_minimum_required(VERSION 3.16)
project(mlib-cuda VERSION 1.0
    DESCRIPTION "Very nice project"
    LANGUAGES CXX CUDA)

#set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
add_definitions(-DWITH_CUDA)

#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CUDA_STANDARD 14)
#set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
#set(CMAKE_CXX_STANDARD_REQUIRED TRUE)



add_library(mlib-cuda "mbm.cu"
    "devmemmanager.cu" "devstreampool.cu"
    "cuda_helpers.cu"  )
set_property(TARGET mlib-cuda PROPERTY CUDA_ARCHITECTURES 50)

find_package( OpenCV 4 REQUIRED)
target_include_directories(mlib-cuda SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(mlib-cuda PUBLIC ${OpenCV_LIBS})
target_link_libraries(mlib-cuda PUBLIC mlib-utils)
add_executable(mbm apps/mbm_main.cpp)
set_property(TARGET mbm PROPERTY CUDA_ARCHITECTURES 50)

target_link_libraries(mbm mlib mlib-cuda)
target_include_directories(mlib-cuda SYSTEM PUBLIC /usr/local/cuda/include)



