cmake_minimum_required(VERSION 3.20)
project(paramgui-app CXX)

if(BUILD_MLIB_APPS)
    file(GLOB apps "*.cpp")
    # add an option for each
    foreach(item ${apps})
        get_filename_component(name ${item} NAME_WE)
        option(build_${name} "compile and link this app" ON)
        if(build_${name})
            add_executable(${name} ${item})
            target_link_libraries(${name} editor)

            #target_include_directories(${name} ${MLIB_TOP_PATH})
            target_configure_warnings(${name})
        endif()
    endforeach()
endif()
