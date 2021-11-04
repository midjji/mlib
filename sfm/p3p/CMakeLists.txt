cmake_minimum_required(VERSION 3.20)

# provide HDRS SRCS
FILE(GLOB HDRS "*.h" "lambdatwist/*.h")
FILE(GLOB SRCS "*.cpp" "lambdatwist/*.cpp")

add_library(mlib-sfm-p3p ${SRCS} ${HDRS})




target_compile_features(mlib-sfm-p3p PUBLIC cxx_std_17)
target_configure_warnings(mlib-sfm-anms)

target_link_libraries(mlib-sfm-p3p PUBLIC param)
target_link_libraries(mlib-sfm-p3p PUBLIC mlib-utils)
target_link_libraries(mlib-sfm-p3p PRIVATE mceres)






