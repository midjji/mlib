cmake_minimum_required(VERSION 3.20)
project(klt-remake VERSION 1.0
    DESCRIPTION "klt remake"
    LANGUAGES CUDA CXX)


file(GLOB_RECURSE internal-sources   "internal/*.h" "internal/*.cpp" "internal/*.cu")
add_library(klt-internal ${internal-sources})
set_property(TARGET klt-internal PROPERTY CUDA_ARCHITECTURES 61)
#target_compile_options(klt-internal PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_60,code=sm_60>)
target_compile_features(klt-internal PUBLIC cxx_std_14)

#target_compile_options(klt-internal PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       #--generate-line-info
                       #--use_fast_math
                       #--relocatable-device-code=true
                       #--forward-unknown-to-host-compiler
                       #>)


get_filename_component(TTTOP_PATH ../../../ REALPATH)
target_include_directories(klt-internal PUBLIC ${TTTOP_PATH})
get_filename_component(TTOP_PATH ../../ REALPATH)
target_include_directories(klt-internal PUBLIC ${TTOP_PATH})
get_filename_component(TOP_PATH ../ REALPATH)
target_include_directories(klt-internal PUBLIC ${TOP_PATH})
target_include_directories(klt-internal SYSTEM PUBLIC "/usr/local/cuda/include")
target_link_libraries(klt-internal PRIVATE mlib-opencv-utils)
target_link_libraries(klt-internal PRIVATE param)
target_link_libraries(klt-internal PRIVATE mlib-utils)


add_library(klt INTERFACE )
target_link_libraries(klt INTERFACE klt-internal)


if(BUILD_MLIB_APPS)
    add_executable(main_klt_tracker "apps/test_klt.cpp")
    target_link_libraries(main_klt_tracker klt)
    target_link_libraries(main_klt_tracker editor)
    target_link_libraries(main_klt_tracker mlib-opencv-utils)
    target_link_libraries(main_klt_tracker mlib-datasets)
    target_link_libraries(main_klt_tracker mlib-utils)

endif()

