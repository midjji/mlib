cmake_minimum_required(VERSION 3.20)
project(param CXX)
file(GLOB srcs  "*.cpp" "*.h" )
add_library(param ${srcs} ) # must be after the qt stuff is set!
target_compile_features(param PUBLIC cxx_std_17)
target_include_directories(param PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(param PRIVATE mlib-utils)

