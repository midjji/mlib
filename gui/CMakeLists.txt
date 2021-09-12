cmake_minimum_required(VERSION 3.16)
project(paramgui CXX)
add_subdirectory(pgui)

add_executable(main main.cpp)
target_link_libraries(main pgui)



