cmake_minimum_required(VERSION 3.20)

# provide HDRS SRCS
FILE(GLOB HDRS "*.h" )
FILE(GLOB SRCS "*.cpp" )

add_library(mlib-sfm-anms ${SRCS} ${HDRS})


target_compile_features(mlib-sfm-anms PUBLIC cxx_std_17)
target_configure_warnings(mlib-sfm-anms)
# We need this directory, and users of our library will need it too
target_link_libraries(mlib-sfm-anms PUBLIC mlib-utils mlib-opencv-utils)

if(build_mlib_tests)
add_subdirectory(test)
endif()
