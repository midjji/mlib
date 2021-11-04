cmake_minimum_required(VERSION 3.20)
project(mlib-gui CXX)
set(CMAKE_AUTOMOC ON)


find_package(Qt5 5.9 REQUIRED Core Gui Widgets )
file(GLOB srcs  "*.cpp" "*.h" )
add_library(mlib-gui ${srcs} ) # must be after the qt stuff is set!

target_link_libraries(mlib-gui  PUBLIC mlib-utils)
target_link_libraries(mlib-gui  PUBLIC mlib-opencv-utils)

target_link_libraries(mlib-gui PUBLIC Qt5::Core Qt5::Gui Qt5::Widgets)




