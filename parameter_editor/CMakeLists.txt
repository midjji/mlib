cmake_minimum_required(VERSION 3.16)
project(paramgui CXX)
set(CMAKE_AUTOMOC ON)


find_package(Qt5 5.9 REQUIRED Core Gui Widgets )
file(GLOB srcs  "*.cpp" "*.h" )
add_library(editor ${srcs} ) # must be after the qt stuff is set!

target_include_directories(editor PUBLIC ${MLIB_TOP_PATH})
target_link_libraries(editor PUBLIC param)
target_compile_features(editor PUBLIC cxx_std_17)

target_include_directories(editor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(editor PUBLIC Qt5::Core Qt5::Gui Qt5::Widgets)

add_subdirectory(apps)


