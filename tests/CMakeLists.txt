cmake_minimum_required(VERSION 3.20)


option(BUILD_MLIB_TESTS "build internal module tests" ${BUILD_MLIB_APPS_DEFAULT})
if(BUILD_MLIB_TESTS)

#testit(matrix "mlib;doctest")
#testit(triangulate_nl "mlib;doctest")
#testit(pose "mlib;doctest")
#testit(tensor "mlib;doctest")
#testit(undistort "mlib;doctest")

#testit(serialize "mlib;doctest")
#testit(syncqueue "mlib;doctest")





add_executable(test_include_all test_include_all.cpp)
target_link_libraries(test_include_all mlib)
#target_configure_warnings(test_include_all)
endif()
