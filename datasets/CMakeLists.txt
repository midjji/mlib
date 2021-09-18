cmake_minimum_required(VERSION 3.16)


file(GLOB srcs  "*.h" "*.cpp" )
list(REMOVE_ITEM SRCS "stereo_datasets.*")

add_library(mlib-datasets "stereo_datasets.cpp")
add_library(mlib-datasets-interface ${srcs})
target_link_libraries(mlib-datasets PUBLIC mlib-datasets-interface)
target_compile_features(mlib-datasets-interface PRIVATE cxx_std_17)
target_link_libraries(mlib-datasets-interface PUBLIC mlib-opencv-utils)



target_include_directories(mlib-datasets-interface INTERFACE PRIVATE "${MLIB_TOP_PATH}")

option(WITH_KITTI "build KITTI" ON)
if(WITH_KITTI)
    add_subdirectory(kitti)
    target_link_libraries(mlib-datasets PUBLIC mlib-kitti)
endif(WITH_KITTI)

option(WITH_TUM "build tum" ON)
if(WITH_TUM)
    add_subdirectory(tum)
    target_link_libraries(mlib-datasets PUBLIC mlib-tum)
endif(WITH_TUM)


if(WITH_SQLite)
    option(WITH_Daimler "build daimler dataset" ON)
    if(WITH_Daimler)
        add_subdirectory(daimler)
        target_link_libraries(mlib-datasets PUBLIC mlib-daimler-dataset )
    endif(WITH_Daimler)
endif()

option(WITH_HILTI "build HILTI" ON)
if(WITH_HILTI)
    add_subdirectory(hilti)
    target_link_libraries(mlib-datasets PUBLIC mlib-hilti)
endif()
