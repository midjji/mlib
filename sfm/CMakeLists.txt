cmake_minimum_required(VERSION 3.20)

add_library(mlib-sfm INTERFACE)
option(WITH_SFM "build SFM" ON)
if(WITH_SFM)
    add_subdirectory(anms)
    target_link_libraries(mlib-sfm INTERFACE mlib-sfm-anms)
    add_subdirectory(p3p)
    target_link_libraries(mlib-sfm INTERFACE mlib-sfm-p3p)
    add_subdirectory(camera)
    target_link_libraries(mlib-sfm INTERFACE mlib-sfm-camera)
endif(WITH_SFM)



