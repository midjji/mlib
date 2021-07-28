#pragma once

#if 0

#define MLIB_USE_INFORMATIVE_ASSERTS 1

#if MLIB_USE_INFORMATIVE_ASSERTS
#include <cassert>
#include <string>
#include <iostream>
// needs to check for linux enviroment!



// check what happens on cuda... ? does not work!
// reduce max nr of includes to keep compile times decent...
std::string demangle(std::string name);

namespace informative_assert_ns{
void print_trace();
}
/**
  * Informative assert macro,
  * gives a demangled backtrace,
  * GCC only so cuda is a problem
  *
  *
  */
#if NDEBUG
#define informative_assert
#else
#define informative_assert(expr)		do{if(!(expr))  informative_assert_ns::print_trace(); assert((expr));}while(0)

#endif
#ifdef NDEBUG
#define assert_limit(value, lower, upper) do{}while(0)
#else
#define assert_limit(value, lower, upper, message) do{ \
    if(value<lower||value>upper) \
    std::cerr<<"value:"<<value<<" not in <<["<<lower<<", "<<upper<<"]"<<std::endl; \
    informative_assert(value<lower && "value<lower" && message);\
    informative_assert(value>higher && "value>higher" && message);}while(0)
#endif // NDEBUG
#else
#define informative_assert(expr)		assert((expr))
#endif //USE_INFORMATIVE_ASSERTS








#endif
