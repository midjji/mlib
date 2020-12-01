cmake_minimum_required(VERSION 3.10)
project(mlib-cuda-apps LANGUAGES CUDA CXX)



macro(testitcu name libs)
    if(BUILD_TESTING)
    string(TOUPPER ${name} uname)
    add_executable(test_${name} test_${name}.cu)
    foreach(item ${libs})
        target_link_libraries(test_${name}  ${item})
    endforeach()
    add_test(TEST_${uname} test_${name} COMMAND TargetName)
endif()
endmacro()

testitcu(cvl_matrix mlib-cuda)
testitcu(dmm mlib-cuda)




