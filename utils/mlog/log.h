#pragma once

/* ********************************* FILE ************************************/
/** \file    log.h
 *
 * \brief    This is a simple logger with autodecoration
 *
 * \remark
 * - autodecoration, i.e. adds function signature,
 * - thread safe and synchronized, i.e. atomic log messages.
 * - to std::cout and to file by default
 * - steady timestamps
 *
 * \Example:
 * mlog()<<"text"<<aclass<<..<<endl;
 * logs to cout and to log.txt,
 * cout lacks decoraction by the default, log.txt has it.
 *
 * also supports:
 * printf(mlog(), const char* format, ...)
 *
 * Issues:
 * - backed by global singletons, the destructors of other global signletons cannot use this class without deconstruction order issues.
 *
 *
 * \remark
 * - c++11
 * - depends on thread
 * - header, cpp
 * - cuda enabled
 * - tested by test_pose.cpp
 *
 * \todo
 * - should implicit homogeneous transforms on multiplication be allowed? yes
 *
 *
 * \author   Mikael Persson
 * \note BSD licence
 *
 ******************************************************************************/
#include <string>
#include <memory>
#include <sstream>
#include <atomic>

static_assert(__cplusplus>=201103L, " must be c++11 or greater");
std::string code_position_impl(std::string function, std::string file, int line);

#define code_position() code_position_impl(std::string(__FILE__),0+__LINE__)

#define mlog() ::cvl::Logger(std::string(__PRETTY_FUNCTION__),std::string(__FILE__),0+__LINE__)
#define mlogl() ::cvl::Logger("",std::string(__FILE__),0+__LINE__)

#define require(condition, message) {\
    if(!(condition)){\
    mlog()<<"\n\nRequired condition: ("<<#condition<<") failed!\n"<<message<<"\n\n";\
    exit(1);}}

#define wtf() require(false, "WTF?\n")

namespace cvl{

struct Tosser{};
template<class T> const Tosser& operator<<(const Tosser& tosser, [[maybe_unused]] T){    return tosser;}


class Log;


class Logger
{

    // should possibly be public ostream, so sprint_f works
public:
    void precision(int digits=6);
    void set_precision_float();
    void set_precision_double();
    void set_precision_long_double();

    // configuration options
    /**
     * @brief set_thread_name sets the name for the calling thread
     * @param name
     *
     */
    void set_thread_name(const std::string& name);
    void set_display_format(bool display_timestamp_,
                            bool display_caller_name_,
                            bool display_file_,
                            bool display_backtrace_){
        display_timestamp=display_timestamp_;
        display_caller_name=display_caller_name_;
        display_file=display_file_;
        display_backtrace=display_backtrace_;
    }


    // things that make it work, but dont worry about it.
    //Just use mlog()<<... as you would have std::cout
    ~Logger();
    Logger(std::string function_description,
           std::string file, unsigned int line,
           std::string log_name="");

    // dealing with stream control types
    // this is the type of std::cout
    using  CoutType = std::basic_ostream<char, std::char_traits<char> >;
    // this is the function signature of std::endl, and most of the other iomanip
    // how do I know which one it is? I dont, instead add them to ss as usual. endl will become "\n" though
    // meaning we will add a flush to the end if any iomanip has been added.
    // A difference from std::cout is that formatting will now be line local. Given that this is for multithreaded enviroments,
    // thats a good thing.
    typedef CoutType& (*StandardEndLine)(CoutType&);

    // define an operator<< to take in std::endl
    const Logger& operator<<(StandardEndLine manip) const;



    // a very rare exception to the the never use mutable rule.
    // Forced by the const reference returned to a temporary variable.
    // using the const reference like this is also strictly not undefined behaviour, but it certainly uncomfortably close.
    // take extreme care when modifying this class.    
    mutable std::stringstream ss;
    mutable bool flush=false;
    static std::atomic<int> digits;
private:
    long int time_ns=0;
    bool display_timestamp=true;
    bool display_caller_name=true;
    bool display_file=true;
    bool display_backtrace=true;
    std::string function_description="";
    std::string file="";
    std::string line="";
    std::shared_ptr<Log> log=nullptr;
};

// a less shit version of string formatting ala printf
template<class ...Args> std::string format_str(std::string format, Args... args){

    // ideally reimplement, but ...   std::format in c++20
    constexpr uint len=2048;
    char buff[len];
    int cx=snprintf( buff, len, format.c_str(), args... );
    if(cx<0)
        return "format parsing failure";
    std::string tmp;tmp.reserve(len);
    for(int i=0;i<cx;++i)
        tmp.push_back(buff[i]);
    return tmp;
    // buffer size is always atleast the size of the format.
}

template<class T>
/**
 * @brief operator <<
 * @param wrapper
 * @param t
 * @return
 *
 * the wrapper is only generated by mlog(), in order to not spam function descriptions when using it as
 * mlog()<<"a"<<"b"<<c<<endl;
 * the function description etc is cleared after the first.
 *
 * What if this throws an exception? Simply put, it mustnt!
 * stream ops generally communicate using error codes anyways.
 *
 */
const Logger& operator<<(const Logger& logger, const T& t) noexcept{
    logger.ss.precision(int(logger.digits));
    logger.ss<<t;
    return logger;
}

template<class ...Args> void printf(const Logger& logger, std::string format, Args... args) {
    logger.ss<<::cvl::format_str(format,args...);
}
}
