cmake_minimum_required(VERSION 3.16)
project(mlib-ceres-util VERSION 1.0
    DESCRIPTION "Very nice project"
    LANGUAGES CXX)

option(WITH_MLIB_CERES_UTILS "build OPENCV_UTILS" ON)
if(NOT WITH_MLIB_CERES_UTILS)
    add_library(mlib-ceres-util  INTERFACE)
else()
# changes means you need to rerun cmake to update
    file(GLOB utils-srcs  "*.cpp" )
    file(GLOB utils-headers  "*.h" )
    add_library(mlib-ceres-util  ${utils-srcs} ${utils-headers})
    target_compile_features(mlib-ceres-util PUBLIC cxx_std_17)
    target_configure_warnings(mlib-ceres-util)



    target_link_libraries(mlib-ceres-util PUBLIC mceres)
    target_link_libraries(mlib-ceres-util PRIVATE mlib-utils)

endif()
