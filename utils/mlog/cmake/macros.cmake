# Build configuration
macro(BuildConfig)

    # Change the default build type from Debug to Release

    # The CACHE STRING logic here and elsewhere is needed to force CMake
    # to pay attention to the value of these variables.(for override)
    if(NOT CMAKE_BUILD_TYPE)
        MESSAGE("-- No build type specified; defaulting to CMAKE_BUILD_TYPE=Release.")
        set(CMAKE_BUILD_TYPE Debug CACHE STRING
                "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."  FORCE)
    else()
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            MESSAGE("\n${line}")
            MESSAGE("\n-- Build type: Debug. Performance will be terrible!")
            MESSAGE("-- Add -DCMAKE_BUILD_TYPE=Release to the CMake command line to get an optimized build.")
            MESSAGE("-- Add -DCMAKE_BUILD_TYPE=RelWithDebInfo to the CMake command line to get an faster build with symbols(-g).")
            MESSAGE("\n${line}")
        endif()
    endif()

    list(APPEND CMAKE_CXX_FLAGS "-std=c++11")
add_definitions("-std=c++11")
endmacro()
macro(WarningConfig)
    #todo add compiler specific...
    option(WExtrawarnings "Extra warnings" ON)
    option(WError "Warnings are errors" OFF)
    if(WError)
        set(warningoptions "${warningoptions} -Werror ")
    endif()


    if(CMAKE_COMPILER_IS_GNUCXX)  # GCC


        # GCC is not strict enough by default, so enable most of the warnings. but disable the annoying ones
        if(WExtrawarnings)

            set(warn "${warn}  -Wall")

            #set(warn "${warn} -Wextra")
            #set(warn "${warn} -Wno-unknown-pragmas")
            #set(warn "${warn} -Wno-sign-compare")
            set(warn "${warn} -Wno-unused-parameter")
            # set(warn "${warn} -Wunused-parameter")
            #set(warn "${warn} -Wno-missing-field-initializers")
            #set(warn "${warn} -Wno-unused")
            #set(warn "${warn} -Wno-unused-function")
            #set(warn "${warn} -Wno-unused-label")
            #set(warn "${warn} -Wno-unused-parameter")
            #set(warn "${warn} -Wno-unused-value")
            set(warn "${warn} -Wno-unused-variable")
            #set(warn "${warn} -Wno-unused-but-set-parameter")
            #set(warn "${warn} -Wno-unused-but-set-variable")

            set(warn "${warn} -Wno-variadic-macros" )
            set(warn "${warn} -Wno-deprecated-declarations" )

            #set(warn "${warn} -Wformat=2 ")
            #set(warn "${warn} -Wnounreachable-code")
            #set(warn "${warn} -Wswitch-default ")
            #set(warn "${warn}     -Winline ")
                #not relevant...
            #set(warn "${warn} -Wshadow")
            #set(warn "${warn} -Weffc++")
            set(warn "${warn} -Wstrict-aliasing")
            set(warn "${warn} -std=c++11") # deprecated flag! should be c++11, but actually we kinda use later and also gnu...
            set(warn "${warn} -pedantic")

        endif()
    endif()


            set(warn "${warn} -Wall -Wextra")
            #set(warn "${warn} -Wno-deprecated-register" )

            set(warningoptions "${warningoptions} ${warn}")
            # disable annoying ones ...
            list(APPEND warningoptions )



        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}${warningoptions}")
        # also no "and" or "or" ! for msvc
        #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-operator-names")
endmacro()
macro(OptimizationConfig)
    if(CMAKE_COMPILER_IS_GNUCXX)  # GCC
        set(CMAKE_CXX_FLAGS_DEBUG "-fno-omit-frame-pointer -pg -g -rdynamic " CACHE STRING "Fixed" FORCE) # dynamic is for the improved asserts
        set(CMAKE_C_FLAGS_DEBUG "-fno-omit-frame-pointer -pg -g -rdynamic " CACHE STRING "Fixed" FORCE) # dynamic is for the improved asserts
        #set(CMAKE_CXX_FLAGS_RELEASE "-fno-rtti -fno-omit-frame-pointer -pg -march=native -mtune=native -O3 -flto -DNDEBUG")
        #set(CMAKE_CXX_FLAGS_RELEASE " -march=native -Ofast -DNDEBUG" )
        set(CMAKE_CXX_FLAGS_RELEASE "-march=native  -O3  -DNDEBUG" CACHE STRING "Fixed" FORCE)
        set(CMAKE_C_FLAGS_RELEASE "-march=native  -O3  -DNDEBUG" CACHE STRING "Fixed" FORCE)
        #set(CMAKE_CXX_FLAGS_RELEASE "-march=native -mtune=native -O2   -DNDEBUG")
        #set(CMAKE_CXX_FLAGS_RELEASE " -O2   -DNDEBUG")
        #set(CMAKE_CXX_FLAGS_RELEASE " -Os   -DNDEBUG")
        #set(CMAKE_CXX_FLAGS_RELEASE " -O2 -Ofast   -DNDEBUG")
        ##set(CMAKE_CXX_FLAGS_RELEASE "-march=native -O2  -DNDEBUG")
    else()
        message("TODO: fix opt options on this compiler")
        set(CMAKE_CXX_FLAGS_RELEASE " -O2 -DNDEBUG")
    endif()
endmacro()

macro(print_list name list)
message("${name}")
foreach(item IN LISTS ${list})
message("     ${item}")
endforeach()
endmacro()

macro(print_filenames name list)
    message("${name}")
    foreach(item IN LISTS ${list})
        get_filename_component(filename ${item} NAME)
        message("     ${filename}")
    endforeach()    
endmacro()

macro(display_library name includes libs) # maybe add defines
    message("${name}")
    set(tab "    ")
    message("${tab}includes:")
    foreach(item IN LISTS ${includes})
        message("${tab}${tab}${item}")
    endforeach()
    message("${tab}libs:")
    foreach(item IN LISTS ${libs})
        message("${tab}${tab}${item}")
    endforeach()
    message("")
endmacro()

