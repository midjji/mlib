cmake_minimum_required(VERSION 3.16)
project(paramhandling VERSION 1.0
    DESCRIPTION "klt remake"
    LANGUAGES CUDA)



file(GLOB_RECURSE sources  "*.h" "*.cpp" "*.cu" "internal/*.h" "internal/*.cpp" "internal/*.cu")
add_library(klt ${sources})

set_property(TARGET klt PROPERTY CUDA_ARCHITECTURES 61)
target_compile_features(klt PUBLIC cxx_std_14)
get_filename_component(TTTOP_PATH ../../../ REALPATH)
target_include_directories(klt PRIVATE ${TTTOP_PATH})
get_filename_component(TTOP_PATH ../../ REALPATH)
target_include_directories(klt PRIVATE ${TTOP_PATH})
get_filename_component(TOP_PATH ../ REALPATH)
target_include_directories(klt PRIVATE ${TOP_PATH})
target_include_directories(klt SYSTEM PRIVATE "/usr/local/cuda/include")
target_link_libraries(klt PRIVATE mlib-opencv-utils)
target_link_libraries(klt PRIVATE param)





find_package(Qt5 5.9 REQUIRED Core Gui Widgets )
add_executable(test_klt "apps/test_klt.cpp")
target_link_libraries(test_klt klt)
target_link_libraries(test_klt editor)
target_link_libraries(test_klt mlib-opencv-utils)
target_link_libraries(test_klt mlib-datasets)

