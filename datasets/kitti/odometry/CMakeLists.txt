cmake_minimum_required(VERSION 3.20)
project(mlib-kitti-odometry VERSION 1.0
    DESCRIPTION "Very nice project"
    LANGUAGES CXX)

# module requires osg, opencv

# provide HDRS SRCS
FILE(GLOB HDRS "*.h")
FILE(GLOB SRCS "*.cpp")
list(REMOVE_ITEM SRCS inspect_kitti_odometry.cpp)
add_library(${PROJECT_NAME} ${SRCS} ${HDRS})

get_filename_component(${PROJECT_NAME}_top ../../ ABSOLUTE)
target_include_directories(${PROJECT_NAME} PUBLIC ${${PROJECT_NAME}_top})

# find opencv and add it
find_package( OpenCV 4 REQUIRED)
target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} PUBLIC mlib-utils)
target_link_libraries(${PROJECT_NAME} PUBLIC mlib-datasets-interface)


target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
target_configure_warnings(${PROJECT_NAME})



