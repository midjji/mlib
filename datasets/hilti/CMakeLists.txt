cmake_minimum_required(VERSION 3.20)
project(mlib-hilti VERSION 1.0
    DESCRIPTION "Very nice project"
    LANGUAGES CXX)

# module requires osg, opencv
# provide HDRS SRCS
FILE(GLOB HDRS "*.h")
FILE(GLOB SRCS "*.cpp")

add_library(${PROJECT_NAME} ${SRCS} ${HDRS})

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

target_link_libraries(${PROJECT_NAME} PUBLIC mlib-opencv-utils)
target_link_libraries(${PROJECT_NAME} PUBLIC mlib-datasets-interface)

target_configure_warnings(${PROJECT_NAME})


add_subdirectory(apps)




