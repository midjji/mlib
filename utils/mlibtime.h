#pragma once
/* ********************************* FILE ************************************/
/** \file    mlibtime.h
 *
 * \brief    This header contains convenience functions for time,
 * timer and date and sleep related functions in a os independent way.
 *
 *
 * \remark
 * - c++11
 * - self contained(just .h,.cpp
 * - no dependencies
 * - os independent works in linux, windows etc are untested for some time
 * - high precision when available
 *
 *
 * It also provides os independent sleep functions
 * Dates converts from system long int ns to isodate according to system clock
 * Convenient classes for keeping track of time
 * The Time class is a wrapper for the std::chrono stuff
 * The Timer class gives you the nice tic and toc functions
 * @code
 * Example:
 * #include <mlib/utils/mlibtime.h>
 *
 * int main(){
 *  Timer timer;
 * float128 seconds=1.5;
 *  for(int i=0;i<10;++i){
 *      timer.tic();
 *      sleep(seconds)
 *      timer.toc();
 *  }
 *  std::cout<< timer<<std::endl;; // gives nice formatted understandable statistics about the time xxx took
 *  // ticks: 1000, mean 400us, median 380us, min 0.0001us max 40000us
 * return 0;
 * }
 * @endcode

 * Timer uses steady clock if available which means correct
 * but slow time calls accuracy is likely in the 5-50ns range, guaranteed to be us
 *
 *
 * \author   Mikael Persson
 * \date     2004-10-01
 * \note MIT licence
 *
 *
 ******************************************************************************/

#include <chrono>
#include <vector>
#include <iostream>
#include <map>


// fits int64 and uint64, this is what should have been used for std::chrono, if not a templated variable size
using float128 = long double;


namespace mlib{



/// Trivial rough time since post stack allocation of the program.
float128 get_steady_now();
/**
 * @brief IsoTime
 * @return the current time as in iso format 24:60:60
 */
std::string getIsoTime();
/**
 * @brief IsoDate
 * @return the current date according to the os in iso format 2016-10-01
 */
std::string getIsoDate();
/**
 * @brief IsoDateTime
 * @return isodate:isotime
 *
 * This s ridiculously complicated very quickly,
 *  so just assume its the time for the local computer
 */
std::string getIsoDateTime();
/**
 * @brief NospaceIsoDateTime
 * @return isodate:isotime
 */
std::string getNospaceIsoDateTime();
/**
 * @brief NospaceIsoDateTime
 * @return isodate:isotime at start of program(within second precision)
 * This function is thread safe, and will guarantee the same answer every time it is called.
 */
std::string getNospaceIsoDateTimeofStart();

/**
 * @brief The Time class
 * A simplified convenient way to manage the std::chrono time system
 */
class Time{
public:
    /// nanoseconds, nothing provides lower than that with any accuracy
    /// float128 covers all of uint64_t, int64_t and eliminates all problems with sign
    /// overflow also behaves more reasonably
    /// the downside is that it is slightly slower, but not much
    float128 ns;
    Time()=default;
    /// from nano seconds @param ns
    Time(float128 ns):ns(ns){}

    /**
     * @brief toStr
     * @return a human readable string representation
     */
    std::string str() const;
    Time& operator+=(Time rhs);
    Time& operator/=(Time rhs);
    float128 seconds() const;
    float128 milli_seconds() const;
};
bool operator==(Time lhs, Time rhs);
bool operator!=(Time lhs, Time rhs);
bool operator< (Time lhs, Time rhs);
bool operator> (Time lhs, Time rhs);
bool operator<=(Time lhs, Time rhs);
bool operator>=(Time lhs, Time rhs);
Time operator+ (Time lhs, Time rhs);
Time operator- (Time lhs, Time rhs);

std::ostream& operator<<(std::ostream &os, Time t);

/**
 * @brief sleep makes this_thread sleep for
 * @param seconds
 */
void sleep(float128 seconds);
/**
 * @brief sleep makes this_thread sleep for
 * @param milliseconds
 */
void sleep_ms(float128 milliseconds);
/**
 * @brief sleep_us makes this_thread sleep for
 * @param microseconds
 */
void sleep_us(float128 microseconds);




class Timer;
/**
 * @brief The TimeScope struct
 */
struct TimeScope{
    Timer* timer; // does not take ownership!
    TimeScope(Timer* timer);
    ~TimeScope();
};




/**
 * @brief The Timer class
 * High precision Timer
 *
 * Timing is accurate in us to ms range
 * 
 */
class Timer{

    /// the last tic
    std::chrono::time_point<std::chrono::steady_clock,std::chrono::nanoseconds > mark;
    /// the last toc
    std::chrono::time_point<std::chrono::steady_clock,std::chrono::nanoseconds > recall;
    /// the time deltas
    std::string name="unnamed";
    bool dotic=true;
    std::vector<Time> ts;




public:
    bool tickable() const;
    bool tockable() const;

    Timer();
    Timer(std::string name, uint capacity=1024);
    Timer(const std::string& name,
          const std::vector<Time>& ts);
    void reserve(uint initialsize);
    /**
     * @brief tic mark the beginning of a new time delta
     */
    void tic();
    /**
     * @brief toc mark the end of the last time delta. if called prior to any toc behaviour is undefined.
     * @return Time in nanoseconds
     */
    Time toc();
    void toss_warmup();


    /**
     * @brief time_scope
     * @return
     *
     * create an instance in a function to tic on creation and toc on destruction, i.e for functions with multiple return paths or exceptions.
     * need to guarantee copy elision!
     * it uh almost guaranteed? yes clang gcc msvc etc all do since 1981
     * yeah, but ugly as sin...
     */
    TimeScope time_scope();
    //TimeScope<Timer>&& time_scope(){return std::forward<TimeScope<Timer>>(TimeScope<Timer>(this));    }


    std::string str() const;
    std::vector<std::string> str_row() const;

    /**
     * @brief clear clears the vector with time deltas
     */
    void clear();
    /**
     * @brief Times
     * @return the time deltas stored so far
     */
    std::vector<Time> times() const;       // alias for Time
    uint64_t samples() const;
    /**
     * @brief Sum
     * @return the sum
     */
    Time sum() const;
    /**
     * @brief Median
     * @return the median
     */
    Time median() const;
    /**
     * @brief Mean
     * @return the mean
     */
    Time mean() const;
    /**
     * @brief Max
     * @return the max
     */
    Time max() const;
    /**
     * @brief Min
     * @return the min
     */
    Time min() const;
};





/**
 * @brief The ScopedDelay struct
 *
 * {ScopedDelay sd(10*1e9);
 * dostuff...
 * }
 * // will take a minimum of 10 seconds,
 * // it will do the thing, then wait.
 *
 * it is equivalent to:
 *
 * end=now()+min_time;
 * dostuff
 * sleep_untill(end);
 * except it still waits even if dostuff throws an exception
 *
 *
 *
 */
struct ScopedDelay{
    std::chrono::time_point<std::chrono::steady_clock,std::chrono::nanoseconds > mark;

    // note if this was uint, the user would need to check how long it has been
    ScopedDelay(float128 delay_ns);
    ~ScopedDelay();
};

class NamedTimerPack{
public:
    std::map<std::string, Timer> ts;
    Timer& make_or_get(const std::string& name);
    Timer& operator[](const std::string& name);


    // have a significant penalty in time due to lookup
    // get the timer reference first instead
    void tic(const std::string& name);
    void toc(const std::string& name);
    std::map<std::string, std::vector<Time>> times();
};

std::ostream& operator<<(std::ostream &os, const std::map<std::string, std::vector<Time>>& ntp);

std::ostream& operator<<(std::ostream &os, const NamedTimerPack& ntp);

/**
 * @brief operator << human readable timer information
 * @param os
 * @param t
 * @return
 */
std::ostream& operator<<(std::ostream &os, const Timer& t);
/**
 * @brief operator << nice human readable timer table
 * @param os
 * @param t
 * @return
 */
std::ostream& operator<<(std::ostream &os, const std::vector<Timer>& t);

}// end namespace mlib
