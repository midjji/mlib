cmake_minimum_required(VERSION 3.16)
project(paramgui CXX)
set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON) # not needed

find_package(Qt5 5.9 REQUIRED Core Gui Widgets )
file(GLOB srcs  "*.cpp" "*.h" )
add_library(pgui ${srcs} ) # must be after the qt stuff is set!
target_compile_features(pgui PUBLIC cxx_std_17)

target_include_directories(pgui PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}) 
target_link_libraries(pgui PUBLIC Qt5::Core Qt5::Gui Qt5::Widgets) 
