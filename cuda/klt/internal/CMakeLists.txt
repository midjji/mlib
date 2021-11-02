project(paramhandling VERSION 1.0
    DESCRIPTION "klt remake utils"
    LANGUAGES CUDA)

file(GLOB_RECURSE sources "*.h" "*.cu" "*.cpp" )

add_library(cudabase ${sources})
set_property(TARGET cudabase PROPERTY CUDA_ARCHITECTURES 61)
target_compile_features(cudabase  PRIVATE cxx_std_14)
get_filename_component(TTOP_PATH ../../ REALPATH)
target_include_directories(cudabase SYSTEM PRIVATE ${TTOP_PATH})
target_include_directories(cudabase SYSTEM PUBLIC "/usr/local/cuda/include")

#set_property(TARGET cudabase PROPERTY CUDA_ARCHITECTURES 50)
