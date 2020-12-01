cmake_minimum_required(VERSION 3.16)
testit(anms "mlib-sfm-anms;doctest")
target_link_libraries(test_anms mlib-utils)
