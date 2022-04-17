file(GLOB srcs "*.cpp")
add_executable(rt main.cc ${srcs})
target_link_libraries(rt mlib)
