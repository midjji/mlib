cmake_minimum_required(VERSION 3.20)

project(mdlidar VERSION 2.0
    DESCRIPTION "read the mdlidar dataset"
    LANGUAGES CXX )

ADD_SUBDIRECTORY(proto)

add_library(mlib-mdlidar read_lidar.cpp)
target_link_libraries(mlib-mdlidar mdlidarproto)
target_link_libraries(mlib-mdlidar mlib-utils)

 
add_executable(read_state test_read_state.cpp )
target_link_libraries(read_state mlib-mdlidar)

add_executable(read_lidar test_read_lidar.cpp )
target_link_libraries(read_lidar mlib-mdlidar)
