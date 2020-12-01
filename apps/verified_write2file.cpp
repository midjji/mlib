#if 1
#include <iostream>
#include <mlib/utils/serialization.h>
#include <fstream>
using namespace mlib;

int main(int argc, char** argv){

    if(argc!=3){
        std::cout<< "make -j8 && ./verified_write infile.txt outfile.txt"<<std::endl;
        return 0;
    }
    std::string str(argv[1]);
    std::ifstream ifs(str);
    if(!ifs){
        std::cout<<"failed to read infile: "<<str;
        return 1;
    }
    std::stringstream ss;
    ss<<ifs.rdbuf();
    std::string path(argv[2]);

    return !mlib::verified_write(ss.str(),path);
}
#else
// A compilation of the following posts:
// http://stackoverflow.com/questions/18648069/g-doesnt-compile-constexpr-function-with-assert-in-it
// http://ericniebler.com/2014/09/27/assert-and-constexpr-in-cxx11/
#include <cassert>
#include <utility>
#include <iostream>

template<class Assert>
inline void constexpr_assert_failed(Assert&& a) noexcept { std::forward<Assert>(a)(); }

// When evaluated at compile time emits a compilation error if condition is not true.
// Invokes the standard assert at run time.
#define constexpr_assert(cond) ((cond) ? 0 : (constexpr_assert_failed([](){ assert(!#cond);}), 0))

/////////////////////////////////////////////////////////////////////
// Usage example:

inline constexpr int divide(int x, int y) noexcept
{
    return constexpr_assert(x <2), x / y;
}


int main()
{

    return assert(min <= max), min <= val && val <= max;

    constexpr auto a = divide(6, 2); // OK
    constexpr auto b = divide(5, 2); // Compile time error in both debug and release builds

    auto a1 = divide(6, 2); // OK
    auto a2 = divide(5, 2); // Run time assertion (debug build only)

    std::cout<<a<<a1<<a2<<std::endl;
    return 0;
}
#endif
