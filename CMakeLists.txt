cmake_minimum_required(VERSION 3.20)

project(mlib VERSION 2.0
    DESCRIPTION "general utility library"
    LANGUAGES CXX CUDA)
message("\n\nmlib configuration begin\n\n")
# everything at once
add_library(mlib INTERFACE )
#set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)




find_package (Threads)
target_link_libraries (mlib INTERFACE Threads::Threads)


#preconfig, also sets
#BUILD_MLIB_APPS_DEFAULT which is true if mlib is
# and
#MLIB_TOP_PATH which is ../, but as a hard path, which is better for ides
include("mlib.cmake")


INIT_BUILD()



# a bunch of different external stuff
add_subdirectory(extern)

#if ( TARGET Qt5::Core )
#        get_property( core_options TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS )
#        string( REPLACE "-fPIC" "" new_core_options "${core_options}" )
#        set_property( TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS ${new_core_options} )
#        set_property( TARGET Qt5::Core PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE "ON" )
#        set( CMAKE_CXX_COMPILE_OPTIONS_PIE "-fPIC" )
#endif()


# dependes on c++11
add_subdirectory(utils)
target_link_libraries(mlib INTERFACE mlib-utils)
# crossdependency on header only stream, stream belongs in utils // TODO

add_subdirectory(stream)
target_link_libraries(mlib INTERFACE mlib-stream)

#depends on ceres publicly
add_subdirectory(ceres_util)
target_link_libraries(mlib INTERFACE mlib-ceres-util)

#depends on opencv
add_subdirectory(opencv_util)
target_link_libraries(mlib INTERFACE mlib-opencv-utils)

add_subdirectory(sfm)
target_link_libraries(mlib INTERFACE mlib-sfm)



add_subdirectory(datasets)
target_link_libraries(mlib INTERFACE mlib-datasets)

# a c++11 only parameter gui
add_subdirectory(param)
target_link_libraries(mlib INTERFACE param)

add_subdirectory(gui)
target_link_libraries(mlib INTERFACE mlib-gui)

# a qt based parameter editor
add_subdirectory(parameter_editor)
target_link_libraries(mlib INTERFACE editor)




# set in extern
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

option(WITH_MLIB_CUDA "build mlib-cuda" ON)
if(WITH_MLIB_CUDA)
    add_subdirectory(cuda)
    target_link_libraries(mlib INTERFACE mlib-cuda)
endif()




# these only get built if mlib isnt a dependency by default
add_subdirectory(apps)
if(build_mlib_tests)
    include(CTest)
endif()
add_subdirectory(tests)

#add_executable(doctest_common_main doctest_common_main.cpp) # must be somewhere once atleast!
#target_link_libraries(doctest_common_main mlib)
message("\n\nmlib configuration complete!\n\n")
