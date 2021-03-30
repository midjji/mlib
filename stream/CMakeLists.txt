cmake_minimum_required(VERSION 3.16)
project(mlib-stream VERSION 1.0
    DESCRIPTION "Very nice project"
    LANGUAGES CXX)
# changes means you need to rerun cmake to update

file(GLOB stream-srcs  "*.cpp" )
#PrintList("${utils-srcs}" "")
file(GLOB stream-headers "*.h" )

add_library(${PROJECT_NAME} ${stream-srcs} ${stream-headers})
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
target_configure_warnings(${PROJECT_NAME})

# We need this directory, and users of our library will need it too, do not use .. in paths, it confuses ides
target_include_directories(${PROJECT_NAME} PUBLIC "${MLIB_TOP_PATH}")




