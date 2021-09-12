cmake_minimum_required(VERSION 3.16)
project(mlib-kitti VERSION 1.0
    DESCRIPTION "Very nice project"
    LANGUAGES CXX)

add_library(mlib-kitti INTERFACE)
add_subdirectory(odometry)
target_link_libraries(mlib-kitti INTERFACE mlib-kitti-odometry)
add_subdirectory(mots)
target_link_libraries(mlib-kitti INTERFACE mlib-kitti-mots)
