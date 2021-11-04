cmake_minimum_required(VERSION 3.20)
project(mtgui CXX)
# changes means you need to rerun cmake to update


#qt5
# Find includes in corresponding build directories


find_package(Qt5 5.9 REQUIRED Core Gui Widgets )

add_library(mtgui "mtgui.h" "mtgui.cpp" ) # must be after the qt stuff is set!
target_compile_features(mtgui PUBLIC cxx_std_17)

target_include_directories(mtgui PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mtgui PRIVATE Qt5::Core Qt5::Gui Qt5::Widgets)


if(${BUILD_MLIB_APPS})
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_INCLUDE_CURRENT_DIR ON) # not needed
    add_executable(main main.cpp)
    target_link_libraries(main mtgui)
    target_link_libraries(main Qt5::Core Qt5::Gui Qt5::Widgets)

    find_package (Threads)
    target_link_libraries (main Threads::Threads)
endif()




