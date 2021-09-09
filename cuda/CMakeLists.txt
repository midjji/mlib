cmake_minimum_required(VERSION 3.20)
project(mlib-cuda VERSION 1.0
    DESCRIPTION "Very nice project"
    LANGUAGES CUDA)


add_subdirectory(cuda-api-wrappers)
add_subdirectory(cuda-kat)
#set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
add_definitions(-DWITH_CUDA)



#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CUDA_STANDARD 14)
#set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
#set(CMAKE_CXX_STANDARD_REQUIRED TRUE)



add_library(mlib-cuda "mbm.cu"
    "devmemmanager.cu" "devstreampool.cu"
    "cuda_helpers.cu"  "cuda_helpers.h")
set_property(TARGET mlib-cuda PROPERTY CUDA_ARCHITECTURES 61)
target_include_directories(mlib-cuda SYSTEM PUBLIC ${MLIB_TOP_PATH})
target_include_directories(mlib-cuda SYSTEM PUBLIC /usr/local/cuda/include)


find_package( OpenCV 4 REQUIRED)
target_include_directories(mlib-cuda SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(mlib-cuda PUBLIC ${OpenCV_LIBS})
#target_link_libraries(mlib-cuda PUBLIC mlib-utils)
add_executable(mbm apps/mbm_main.cpp)
add_executable(add apps/add.cu)
set_property(TARGET add PROPERTY CUDA_ARCHITECTURES 61)
target_link_libraries(add runtime-api)
target_link_libraries(add cuda-kat)

#set_property(TARGET mbm PROPERTY CUDA_ARCHITECTURES 61)

target_link_libraries(add mlib mlib-cuda)
target_link_libraries(mbm mlib mlib-cuda)
target_include_directories(mbm SYSTEM PUBLIC /usr/local/cuda/include)



