cmake_minimum_required(VERSION 3.16)

option(BUILD_CERES "build CERES" ON)
if(BUILD_CERES)
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ceres-solver/CMakeLists.txt")
    message(FATAL_ERROR "ceres-solver not found, clone it here: ${CMAKE_CURRENT_SOURCE_DIR}/")
endif()


# set a bunch of defaults, then clear them up
mark_as_advanced(FORCE BUILD_DOCUMENTATION)
mark_as_advanced(FORCE BUILD_BENCHMARKS)
mark_as_advanced(FORCE BUILD_SHARED_LIBS)
mark_as_advanced(FORCE CERES_THREADING_MODEL)
mark_as_advanced(FORCE CUSTOM_BLAS)
mark_as_advanced(FORCE CXSPARSE)
mark_as_advanced(FORCE EIGENSPARSE)
mark_as_advanced(FORCE EIGEN3_DIR)
mark_as_advanced(FORCE GFLAGS)
mark_as_advanced(FORCE gflags_DIR)
mark_as_advanced(FORCE LAPACK)
mark_as_advanced(FORCE MINIGLOG)
mark_as_advanced(FORCE BUILD_EXAMPLES)
mark_as_advanced(FORCE PROVIDE_UNINSTALL_TARGET)
mark_as_advanced(FORCE CCACHE_PROGRAM)
mark_as_advanced(FORCE SUITESPARSE)
mark_as_advanced(FORCE LIB_SUFFIX)
mark_as_advanced(FORCE SCHUR_SPECIALIZATIONS)
mark_as_advanced(FORCE SANITIZERS)


set(CMAKE_CXX_STANDARD 17)
SET(BUILD_EXAMPLES OFF CACHE BOOL "") # speed
SET(BUILD_TESTING OFF CACHE BOOL "") # speed
SET(EXPORT_BUILD_DIR ON CACHE BOOL "") # critical!
add_subdirectory(ceres-solver)


else()
    # dont use internal ceres for debugging
     find_package(Ceres 2 REQUIRED)

endif()

