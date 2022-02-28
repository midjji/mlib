cmake_minimum_required(VERSION 3.20)
project(mlib-utils VERSION 1.0
    DESCRIPTION "Very nice project"
    LANGUAGES CXX)
# changes means you need to rerun cmake to update

file(GLOB utils-srcs  "*.cpp" "mlog/log.cpp" "spline/*.cpp")
#PrintList("${utils-srcs}" "")
file(GLOB utils-headers  "*.h" "*.hpp" "cvl/*.h" "cvl/*.hpp" "mzip/*.h" "spline/*.h")

add_library(${PROJECT_NAME} ${utils-srcs} ${utils-headers})
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

target_configure_warnings(${PROJECT_NAME})

# We need this directory, and users of our library will need it too
target_include_directories(${PROJECT_NAME} PUBLIC "${MLIB_TOP_PATH}")

# IDEs should put the headers in a nice place
#source_group(TREE "${PROJECT_SOURCE_DIR}/include" PREFIX "Header Files" FILES ${utils-headers})
FIND_FILESYSTEM()
target_link_libraries(${PROJECT_NAME} PUBLIC stdfs )


#unfortunately the quaternion requires ceres, but this can be trivially removed if you dont plan to use them in optimization.
target_link_libraries(mlib-utils PRIVATE mlib-extern)

# this is required for __host__ __device__
target_include_directories(mlib-utils PUBLIC "/usr/local/cuda/include/")



