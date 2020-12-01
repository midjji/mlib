cmake_minimum_required(VERSION 3.10)
project(mlib-kitti VERSION 1.0
    DESCRIPTION "Very nice project"
    LANGUAGES CXX)

include_directories(".")
add_subdirectory(odometry)
add_subdirectory(mots)

add_library(mlib-kitti INTERFACE)
target_link_libraries(mlib-kitti INTERFACE mlib-kitti-odometry)
target_link_libraries(mlib-kitti INTERFACE mlib-kitti-mots)
