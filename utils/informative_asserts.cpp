//#if MLIB_USE_INFORMATIVE_ASSERTS

#include <execinfo.h>
#include <cstdio>
#include <cstdlib>
#include <cxxabi.h>
#include <string>
#include <iostream>


std::string demangle(std::string name){

    // assumes output from backtrace_symbols ./,,,(name)[]

    std::size_t start=name.find_first_of("(");
    std::size_t end=name.find_first_of("+");
    if(start==std::string::npos || end==std::string::npos||start+1>end-1) return "demangle failure: "+name;

    name=name.substr(start+1,end-start-1); // cuts out the first part
    if(name=="main") return "main()";
    if(name=="__libc_start_main") return name;




    int status = -1;
    //cout<<"demangle: "<<name<<endl;
// no way to avoid allocation ... possible error source...
    char* demangledName = abi::__cxa_demangle( name.c_str(), nullptr, nullptr, &status ); // seriously why isnt there a nice non ptr version...
    std::string ret="unknown demangle failure";
    if ( status == 0 )
    {
        if(demangledName !=nullptr)
        ret=demangledName;

    }

    free( demangledName );
    return ret;
}


namespace informative_assert_ns{
/* Obtain a backtrace and print it to stdout. */
void print_trace ()
{
    void *array[5];
    int size;
    char **strings;

    size = backtrace (array, 5);
    strings = backtrace_symbols (array, size);

    std::cout<<"Informative Assert failed:  "<<size<<" stack frames.\n"<<std::endl;

    for (int i = 1; i < size-1; ++i){
        std::cout<<demangle(strings[i])<<std::endl;
    }

    free (strings);
}

}
//#endif //USE_INFORMATIVE_ASSSERTS






