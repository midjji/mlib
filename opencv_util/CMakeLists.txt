cmake_minimum_required(VERSION 3.16)
project(mlib-opencv-utils VERSION 1.0
    DESCRIPTION "Very nice project"
    LANGUAGES CXX)



option(WITH_MLIB_OPENCV_UTILS "build OPENCV_UTILS" ON)
if(NOT WITH_MLIB_OPENCV_UTILS)
    add_library(mlib-opencv-utils INTERFACE)
else()
# changes means you need to rerun cmake to update
file(GLOB utils-srcs  "*.cpp" )
file(GLOB utils-headers  "*.h" )
add_library(mlib-opencv-utils ${utils-srcs} ${utils-headers})
target_compile_features(mlib-opencv-utils PUBLIC cxx_std_17)
target_configure_warnings(mlib-opencv-utils)

# find opencv and add it
find_package( OpenCV 4 REQUIRED)
message("OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
target_include_directories(mlib-opencv-utils SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(mlib-opencv-utils PUBLIC ${OpenCV_LIBS})
target_link_libraries(mlib-opencv-utils PUBLIC mlib-utils)

endif()
