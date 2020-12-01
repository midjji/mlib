cmake_minimum_required(VERSION 3.10)




testit(matrix "mlib;doctest")
#testit(triangulate_nl "mlib;doctest")
testit(pose "mlib;doctest")
#testit(tensor "mlib;doctest")
#testit(undistort "mlib;doctest")

testit(serialize "mlib;doctest")
testit(syncqueue "mlib;doctest")





add_executable(test_include_all test_include_all.cpp)
target_link_libraries(test_include_all mlib)
target_configure_warnings(test_include_all)
