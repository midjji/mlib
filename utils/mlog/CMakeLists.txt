cmake_minimum_required(VERSION 3.5)
project(mlog)
include(cmake/macros.cmake)


BuildConfig()
WarningConfig()
OptimizationConfig()




find_package(Threads REQUIRED)
INCLUDE_DIRECTORIES("../")
add_library(mlog log.h log.cpp)
target_link_libraries(mlog Threads::Threads)
add_executable(mlog_example mlog_example.cpp)
target_link_libraries(mlog_example mlog)





















