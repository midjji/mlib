cmake_minimum_required(VERSION 3.16)
project(mlib-plotter CXX)


add_library(mlib-plotter "plot.h" "plot.cpp" ) # must be after the qt stuff is set!
add_subdirectory(mtgui)
target_link_libraries(mlib-plotter PRIVATE mtgui)
target_link_libraries(mlib-plotter PRIVATE jkqtplotter)
target_link_libraries(mlib-plotter PRIVATE mlib-utils)

FIND_PACKAGE(Threads REQUIRED )
target_link_libraries(mlib-plotter PRIVATE Threads::Threads)





