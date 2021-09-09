cmake_minimum_required(VERSION 3.16)
project(mlib-daimler-dataset VERSION 1.0
    DESCRIPTION "Very nice project"
    LANGUAGES CXX)

# module requires osg, opencv
# provide HDRS SRCS
FILE(GLOB HDRS "*.h")
FILE(GLOB SRCS "*.cpp")

add_library(${PROJECT_NAME} ${SRCS} ${HDRS})


# also needs path to dir above...#
#get_filename_component(${PROJECT_NAME}_top ../ ABSOLUTE)
#target_include_directories(${PROJECT_NAME} PUBLIC ${${PROJECT_NAME}_top})


target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

target_link_libraries(${PROJECT_NAME} PUBLIC mlib-opencv-utils)
target_link_libraries(${PROJECT_NAME} PUBLIC mlib-datasets-interface)

target_configure_warnings(${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME} PUBLIC sqlite_orm)




