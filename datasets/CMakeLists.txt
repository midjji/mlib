cmake_minimum_required(VERSION 3.16)

add_library(mlib-datasets INTERFACE)

file(GLOB srcs  "*.h" "*.cpp" )

add_library(mlib-datasets-interface ${srcs})
target_compile_features(mlib-datasets-interface PRIVATE cxx_std_17)

find_package( OpenCV 4 REQUIRED)
target_link_libraries(mlib-datasets-interface )
message("OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
target_include_directories(mlib-datasets-interface SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(mlib-datasets-interface PUBLIC ${OpenCV_LIBS})



target_include_directories(mlib-datasets-interface INTERFACE PRIVATE "${MLIB_TOP_PATH}")
target_link_libraries(mlib-datasets INTERFACE mlib-datasets-interface)


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
