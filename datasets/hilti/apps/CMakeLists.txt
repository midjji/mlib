cmake_minimum_required(VERSION 3.16)
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_BINARY_DIR}/bin)

option(BUILD_MLIB_APPS "build mlib apps" ${BUILD_MLIB_APPS_DEFAULT})
if(BUILD_MLIB_APPS)
file(GLOB apps "*.cpp" )

# add an option for each
foreach(item ${apps})
    get_filename_component(name ${item} NAME_WE)
    option(build_${name} "compile and link this app" ON)
    if(build_${name})
        add_executable(${name} ${item})
        target_link_libraries(${name} mlib)
        target_configure_warnings(${name})
    endif()
endforeach()

endif()
