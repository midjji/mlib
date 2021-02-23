cmake_minimum_required(VERSION 3.16)

project(mlib VERSION 1.0
    DESCRIPTION "general utility library"
    LANGUAGES CXX)
message("\n\nmlib configuration begin\n\n")

include("extern/cmake/common.cmake")
# add this path to module path, so any find modules here are founds
get_filename_component(MLIB_COMMON_CMAKE_PATH ./ REALPATH)
LIST(APPEND CMAKE_MODULE_PATH "${MLIB_COMMON_CMAKE_PATH}/extern/cmake/" )
LIST(APPEND CMAKE_MODULE_PATH "/usr/local/lib/cmake/" )
print_list(CMAKE_MODULE_PATH CMAKE_MODULE_PATH) # uncomment to list where you are looking
include("extern/cmake/dependencies.cmake")
set(BUILD_MLIB_EXTRAS_DEFAULT ON)
get_directory_property(hasParent PARENT_DIRECTORY)
if(hasParent)
    set(BUILD_MLIB_EXTRAS_DEFAULT OFF)
endif()
option(BUILD_MLIB_APPS "build mlib apps" ${BUILD_MLIB_EXTRAS_DEFAULT})
option(BUILD_MLIB_TESTS "build internal module tests"  ${BUILD_MLIB_EXTRAS_DEFAULT})

add_library(mlib INTERFACE)

# cmake strips duplicate include paths, but it requires the real paths to do so.
# it fails for ../ which messes with autocomplete
get_filename_component(MLIB_TOP_PATH ../ REALPATH)
target_include_directories(mlib INTERFACE "${MLIB_TOP_PATH}")
message("Found MLIB: ${MLIB_TOP_PATH}")
message("CMAKE_MODULE_PATH")
print_list("${CMAKE_MODULE_PATH}" "")


add_subdirectory(extern)
target_link_libraries(mlib INTERFACE mlib-extern)



INIT_BUILD()
if(build_mlib_tests)
    include(CTest)
endif()

add_subdirectory(utils)
target_link_libraries(mlib INTERFACE mlib-utils)
add_subdirectory(sfm)
target_link_libraries(mlib INTERFACE mlib-sfm)


add_subdirectory(opencv_util)
target_link_libraries(mlib INTERFACE mlib-opencv-utils)

add_subdirectory(datasets)
target_link_libraries(mlib INTERFACE mlib-datasets)




if(WITH_JKQTPLOTTER)
    # requires with_MLIB_QT and WITH_JKQTPLOTTER
    add_subdirectory(plotter)
    target_link_libraries(mlib INTERFACE mlib-plotter)
endif(WITH_JKQTPLOTTER)


option(WITH_VIS "build mlib-vis" ON)
if(WITH_VIS)
    add_subdirectory(vis)
    target_link_libraries(mlib INTERFACE mlib-vis)
endif(WITH_VIS)

option(WITH_MLIB_CUDA "build mlib-cuda" OFF)
if(WITH_MLIB_CUDA)
    #add_subdirectory(cuda)
    #target_link_libraries(mlib INTERFACE mlib-cuda)
endif()


if(BUILD_MLIB_APPS)
    add_subdirectory(apps)
endif()
if(BUILD_MLIB_TESTS)
    add_subdirectory(tests)
endif()
add_executable(doctest_common_main doctest_common_main.cpp) # must be somewhere once atleast!
target_link_libraries(doctest_common_main mlib)
message("\n\nmlib configuration complete!\n\n")
