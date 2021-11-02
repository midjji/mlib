cmake_minimum_required(VERSION 3.16)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


option(BUILD_MLIB_APPS "build mlib apps" ${BUILD_MLIB_APPS_DEFAULT})
if(BUILD_MLIB_APPS)
file(GLOB apps "*.cpp" "../utils/mlog/mlog_example.cpp" "../utils/mzip/test_mzip.cpp")

#set(apps "")

#list(APPEND apps "test_stereo.cpp")
#list(APPEND apps "test_undistort.cpp")
#list(APPEND apps "../utils/mlog/mlog_example.cpp")
#list(APPEND apps "../utils/mzip/test_mzip.cpp")
if(WITH_KITTI)
#list(APPEND apps "inspect_kitti_odometry.cpp")
#list(APPEND apps "inspect_kitti_mots.cpp")
#list(APPEND apps "inspect_stereo_dataset.cpp")
endif()
if(WITH_VIS)
#list(APPEND apps "vis_manip.cpp")
#list(APPEND apps "test_lookat.cpp")
endif()
if(WITH_SQLite)
#list(APPEND apps "sqlite.cpp")
endif()
#list(APPEND apps "verified_write2file.cpp")








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



message("apps done\n")

endif()
