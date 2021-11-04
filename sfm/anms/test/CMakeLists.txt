cmake_minimum_required(VERSION 3.20)
testit(anms "mlib-sfm-anms;doctest")
target_link_libraries(test_anms mlib-utils)
