cmake_minimum_required(VERSION 3.16)
project(mlib-vis CXX)



message("Processing Module mlib-vis")
# module requires osg, opencv

# packages


# provide HDRS SRCS
FILE(GLOB HDRS "*.h")
FILE(GLOB SRCS "*.cpp")

add_library(mlib-vis ${SRCS} ${HDRS})
target_configure_warnings(mlib-vis)
target_link_libraries(mlib-vis PUBLIC mlib-utils)
target_link_libraries(mlib-vis PUBLIC mlib-stream)

#find_package( OpenCV 4 REQUIRED)
#target_include_directories(mlib-vis SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})
#target_link_libraries(mlib-vis PUBLIC ${OpenCV_LIBS})


macro(FIND_OSG)
    message("\n${line}\n")
    message("OSG\n")
    find_package(OpenSceneGraph 3 REQUIRED osgDB osgUtil osgViewer osgGA osgWidget REQUIRED)
    message("\nFound OSG Version: ${OPENSCENEGRAPH_VERSION}")

    # hide clutter
    mark_as_advanced(FORCE OPENTHREADS_INCLUDE_DIR)
    mark_as_advanced(FORCE OPENTHREADS_LIBRARY)
    mark_as_advanced(FORCE OPENTHREADS_LIBRARY_DEBUG)
    mark_as_advanced(FORCE OSGDB_INCLUDE_DIR)
    mark_as_advanced(FORCE OSGDB_LIBRARY     )
    mark_as_advanced(FORCE OSGDB_LIBRARY_DEBUG)
    mark_as_advanced(FORCE OSGGA_INCLUDE_DIR   )
    mark_as_advanced(FORCE OSGGA_LIBRARY        )
    mark_as_advanced(FORCE OSGGA_LIBRARY_DEBUG   )
    mark_as_advanced(FORCE OSGUTIL_INCLUDE_DIR )
    mark_as_advanced(FORCE OSGUTIL_LIBRARY      )
    mark_as_advanced(FORCE OSGUTIL_LIBRARY_DEBUG )
    mark_as_advanced(FORCE OSGVIEWER_INCLUDE_DIR  )
    mark_as_advanced(FORCE OSGVIEWER_LIBRARY       )
    mark_as_advanced(FORCE OSGVIEWER_LIBRARY_DEBUG  )
    mark_as_advanced(FORCE OSGWIDGET_INCLUDE_DIR  )
    mark_as_advanced(FORCE OSGWIDGET_LIBRARY )
    mark_as_advanced(FORCE OSGWIDGET_LIBRARY_DEBUG   )
    mark_as_advanced(FORCE OSG_INCLUDE_DIR  )
    mark_as_advanced(FORCE OSG_LIBRARY         )
    mark_as_advanced(FORCE OSG_LIBRARY_DEBUG)


    if(verbose)
        message("OpenSceneGraph include directories: ${OPENSCENEGRAPH_INCLUDE_DIR}")
        message("OpenSceneGraph libraries: ")
        Print_List(OPENSCENEGRAPH_LIBRARIES OPENSCENEGRAPH_LIBRARIES)
    endif()
endmacro()
FIND_OSG()

target_link_libraries(mlib-vis PUBLIC ${OPENSCENEGRAPH_LIBRARIES})
target_include_directories(mlib-vis SYSTEM PUBLIC ${OPENSCENEGRAPH_INCLUDE_DIR})

if(build_mlib_tests)
    add_subdirectory(tests)
endif()


