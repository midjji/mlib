cmake_minimum_required(VERSION 3.10)
# provide HDRS SRCS
FILE(GLOB HDRS "*.h" )
FILE(GLOB SRCS "*.cpp" )

add_library(mlib-sfm-camera ${SRCS} ${HDRS})

