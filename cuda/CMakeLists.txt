cmake_minimum_required(VERSION 3.20)
#set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
project(mlib-cuda VERSION 1.0
    DESCRIPTION "Very nice project"
    LANGUAGES CXX CUDA)


#add_subdirectory(cuda-api-wrappers)
#add_subdirectory(cuda-kat)


#set(CMAKE_CUDA_STANDARD 11)
#set(CMAKE_CUDA_STANDARD_REQUIRED ON)

#set(CMAKE_CUDA_ARCHITECTURES 61)
SET(CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(mlib INTERFACE SYSTEM "/usr/local/cuda/include")


#set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
#set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

add_library(mlib-cuda INTERFACE)
add_subdirectory(klt)
target_link_libraries(mlib-cuda INTERFACE klt)
#add_library(mbml "mbm.cu" "devmemmanager.cu" "devstreampool.cu" "cuda_helpers.cu"  "cuda_helpers.h")
#set_property(TARGET mbml PROPERTY CUDA_ARCHITECTURES 61)
#target_link_libraries(mbml PRIVATE mlib-opencv-utils)
#target_link_libraries(mlib-cuda INTERFACE mbml)


#set_property(TARGET mlib-cuda PROPERTY CUDA_ARCHITECTURES 61)
#target_include_directories(mlib-cuda SYSTEM PUBLIC ${MLIB_TOP_PATH})
#target_include_directories(mlib-cuda SYSTEM PUBLIC /usr/local/cuda/include)


#find_package( OpenCV 4 REQUIRED)
#target_include_directories(mlib-cuda SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})
#target_link_libraries(mlib-cuda PUBLIC ${OpenCV_LIBS})
#target_link_libraries(mlib-cuda INTERFACE mlib-utils)
#add_executable(mbm apps/mbm_main.cpp)

#add_executable(add apps/add.cu)
#set_property(TARGET add PROPERTY CUDA_ARCHITECTURES 61)
#target_link_libraries(add runtime-api)
#target_link_libraries(add cuda-kat)

#set_property(TARGET mbm PROPERTY CUDA_ARCHITECTURES 61)

#target_link_libraries(add mlib mlib-cuda)
#target_link_libraries(mbm mlib mlib-cuda)
#target_include_directories(mbm SYSTEM PUBLIC /usr/local/cuda/include)

add_executable(raytrace apps/raytrace.cu)
set_property(TARGET raytrace PROPERTY CUDA_ARCHITECTURES 61)
target_link_libraries(raytrace mlib mlib-cuda)

