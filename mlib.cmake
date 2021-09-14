# move the messy part to this file so the main cmake is more readable
# common cmake helpers
include("extern/cmake/common.cmake")
# add this path to module path, so any find modules here are founds
get_filename_component(MLIB_COMMON_CMAKE_PATH ./ REALPATH)
LIST(APPEND CMAKE_MODULE_PATH "${MLIB_COMMON_CMAKE_PATH}/extern/cmake/" )
LIST(APPEND CMAKE_MODULE_PATH "/usr/local/lib/cmake/" )
#print_list(CMAKE_MODULE_PATH CMAKE_MODULE_PATH) # uncomment to list where you are looking
include("extern/cmake/dependencies.cmake")

# top level directory
get_filename_component(MLIB_TOP_PATH ../ REALPATH)
target_include_directories(mlib INTERFACE "")
get_directory_property(MLIB_SUBDIR PARENT_DIRECTORY)
set(BUILD_MLIB_APPS_DEFAULT ON)
if(MLIB_SUBDIR)
    set(BUILD_MLIB_APPS_DEFAULT OFF)
endif()
