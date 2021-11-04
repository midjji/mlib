cmake_minimum_required(VERSION 3.20)
project(mlib-tum CXX)



add_library(mlib-tum tum.h tum.cpp)

target_compile_features(mlib-tum PUBLIC cxx_std_17)
WarningConfig()
target_compile_options(mlib-tum PRIVATE ${WARNINGS})



# also needs path to dir above...
get_filename_component(${PROJECT_NAME}_top ../ ABSOLUTE)
target_include_directories(${PROJECT_NAME} PUBLIC ${${PROJECT_NAME}_top})
target_link_libraries(mlib-tum PUBLIC mlib-opencv-utils)





if(build_mlib_apps)
    if(WITH_VIS)
        add_executable(inspect_tum inspect_tum.cpp)
        target_link_libraries(inspect_tum mlib-tum mlib-vis)
        set_target_properties( inspect_tum PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/apps")
    endif()
endif()

