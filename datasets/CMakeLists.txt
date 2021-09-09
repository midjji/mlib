cmake_minimum_required(VERSION 3.16)


file(GLOB srcs  "*.h" "*.cpp" )
add_library(mlib-datasets INTERFACE)
add_library(mlib-datasets-interface ${srcs})
target_compile_features(mlib-datasets-interface PRIVATE cxx_std_17)
target_link_libraries(mlib-datasets-interface PUBLIC mlib-opencv-utils)



target_include_directories(mlib-datasets-interface INTERFACE PRIVATE "${MLIB_TOP_PATH}")

if(WITH_MLIB_OPENCV_UTILS)
option(WITH_KITTI "build KITTI" ON)
if(WITH_KITTI)
    add_subdirectory(kitti)
    target_link_libraries(mlib-datasets PUBLIC INTERFACE mlib-kitti)
endif(WITH_KITTI)

option(WITH_TUM "build tum" ON)
if(WITH_TUM)
    add_subdirectory(tum)
    target_link_libraries(mlib-datasets PUBLIC INTERFACE mlib-tum)
endif(WITH_TUM)


if(WITH_SQLite)
    option(WITH_Daimler "build daimler dataset" ON)
    if(WITH_Daimler)
        add_subdirectory(daimler)
        target_link_libraries(mlib-datasets PUBLIC INTERFACE mlib-daimler-dataset )
    endif(WITH_Daimler)
endif()

endif()

