cmake_minimum_required(VERSION 3.16)
project(paramgui-example CXX)

add_executable(paramgui-example "main.cpp" "a.cpp")
target_link_libraries(paramgui-example editor)

