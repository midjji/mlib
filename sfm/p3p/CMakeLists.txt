cmake_minimum_required(VERSION 3.10)

# provide HDRS SRCS
FILE(GLOB HDRS "*.h" "lambdatwist/*.h")
FILE(GLOB SRCS "*.cpp" "lambdatwist/*.cpp")

add_library(mlib-sfm-p3p ${SRCS} ${HDRS})



target_compile_features(mlib-sfm-p3p PUBLIC cxx_std_11)
target_configure_warnings(mlib-sfm-anms)



target_link_libraries(mlib-sfm-p3p PUBLIC mlib-utils)

find_package(Ceres 2 REQUIRED)
target_include_directories(mlib-sfm-p3p SYSTEM PUBLIC ${CERES_INCLUDE_DIRS})
target_link_libraries(mlib-sfm-p3p PUBLIC ${CERES_LIBRARIES})


