cmake_minimum_required(VERSION 3.16)

add_library(mlib-extern INTERFACE)

# required but external
add_subdirectory(convenient_multithreaded_qt_gui)
add_subdirectory(ceres)
target_link_libraries(mlib-extern INTERFACE ceres)

target_link_libraries(mlib-extern INTERFACE mtgui)


# doctest is only a single header, so others may well have included it.

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/doctest/CMakeLists.txt")
    message(FATAL_ERROR "doctest not found, clone it")
endif()
add_library(doctest INTERFACE)
target_include_directories(doctest SYSTEM INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/doctest/doctest/")
target_link_libraries(mlib-extern INTERFACE doctest)


option(WITH_JKQTPLOTTER "build JKQTPLOTTER" ON)
if(WITH_JKQTPLOTTER)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/JKQtPlotter/CMakeLists.txt")
        message(FATAL_ERROR "use clone_dependencies.sh in this dir\nor git clone git@github.com:midjji/JKQtPlotter.git")
    endif()
    add_subdirectory(JKQtPlotter)
    target_link_libraries(mlib-extern INTERFACE jkqtplotter)
endif()

option(WITH_SQLite "use sqlite_orm" ON)
if(WITH_SQLite)
    #sudo apt install libsqlite3-dev
    #find sqlite only works from cmake3.14
    #    if(NOT EXISTS "/usr/include/sqlite3.h")
    #        message(FATAL_ERROR "Needs sqlite3\n sudo apt install libsqlite3-dev \n Warning, this is ubuntu specific, update the file extern/cmake/FindSqlLite.cmake to where the .h and .so files of sqlite are to fix")
    #    endif()

    add_library(SQLite::SQLite3 UNKNOWN IMPORTED)
    set_target_properties(SQLite::SQLite3 PROPERTIES
        IMPORTED_LOCATION             "/usr/lib/x86_64-linux-gnu/libsqlite3.so"
        INTERFACE_INCLUDE_DIRECTORIES "/usr/include/")


    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/sqlite_orm/CMakeLists.txt")
        message(FATAL_ERROR "git clone https://github.com/fnc12/sqlite_orm.git")
    endif()

    add_library(sqlite_orm INTERFACE)
    target_link_libraries(sqlite_orm INTERFACE SQLite::SQLite3)
    target_include_directories(sqlite_orm INTERFACE SYSTEM "${CMAKE_CURRENT_SOURCE_DIR}/sqlite_orm/include/sqlite_orm")
    target_link_libraries(mlib-extern INTERFACE sqlite_orm)
endif()
