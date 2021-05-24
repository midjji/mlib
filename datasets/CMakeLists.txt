cmake_minimum_required(VERSION 3.16)

add_library(mlib-datasets INTERFACE)

if(WITH_MLIB_OPENCV_UTILS)
option(WITH_KITTI "build KITTI" ON)
if(WITH_KITTI)
    add_subdirectory(kitti)
    target_link_libraries(mlib-datasets INTERFACE mlib-kitti)
endif(WITH_KITTI)

option(WITH_TUM "build tum" ON)
if(WITH_TUM)
    add_subdirectory(tum)
    target_link_libraries(mlib-datasets INTERFACE mlib-tum)
endif(WITH_TUM)


if(WITH_SQLite)
    option(WITH_Daimler "build daimler dataset" ON)
    if(WITH_Daimler)
        add_subdirectory(daimler)
        target_link_libraries(mlib-datasets INTERFACE mlib-daimler-dataset)
    endif(WITH_Daimler)
endif()
endif()
