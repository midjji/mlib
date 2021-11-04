cmake_minimum_required(VERSION 3.20)
project(klt-remake VERSION 1.0
    DESCRIPTION "klt remake"
    LANGUAGES CUDA CXX)


file(GLOB_RECURSE internal-sources   "internal/*.h" "internal/*.cpp" "internal/*.cu")
add_library(klt-internal ${internal-sources})
set_property(TARGET klt-internal PROPERTY CUDA_ARCHITECTURES 61)
target_compile_features(klt-internal PUBLIC cxx_std_17)
get_filename_component(TTTOP_PATH ../../../ REALPATH)
target_include_directories(klt-internal PUBLIC ${TTTOP_PATH})
get_filename_component(TTOP_PATH ../../ REALPATH)
target_include_directories(klt-internal PUBLIC ${TTOP_PATH})
get_filename_component(TOP_PATH ../ REALPATH)
target_include_directories(klt-internal PUBLIC ${TOP_PATH})
target_include_directories(klt-internal SYSTEM PUBLIC "/usr/local/cuda/include")
target_link_libraries(klt-internal PRIVATE mlib-opencv-utils)
target_link_libraries(klt-internal PRIVATE param)

file(GLOB_RECURSE sources  "*.h" "*.cpp")
add_library(klt ${sources})
target_link_libraries(klt PRIVATE klt-internal)



target_link_libraries(klt PRIVATE mlib-opencv-utils)
target_link_libraries(klt PRIVATE param)

if(BUILD_MLIB_APPS)
    add_executable(test_klt "apps/test_klt.cpp")
    target_link_libraries(test_klt klt)
    target_link_libraries(test_klt editor)
    target_link_libraries(test_klt mlib-opencv-utils)
    target_link_libraries(test_klt mlib-datasets)
endif()

