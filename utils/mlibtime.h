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
 * double seconds=1.5;
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

#ifndef Time_H
#define Time_H

#include <chrono>
#include <vector>
#include <iostream>
#include <map>


namespace mlib{



/// different ways of counting time seconds, milliseconds, microseconds, nanoseconds
enum TIME_TYPE{TIME_S,TIME_MS,TIME_US,TIME_NS};


/// Trivial rough time since post stack allocation of the program.
uint64_t get_steady_now();
/**
 * @brief getIsoTime
 * @return the current time as in iso format 24:60:60
 */
std::string getIsoTime();
/**
 * @brief getIsoDate
 * @return the current date according to the os in iso format 2016-10-01
 */
std::string getIsoDate();
/**
 * @brief getIsoDateTime
 * @return isodate:isotime
 *
 * This gets ridiculously complicated very quickly, so just assume its the time for the local computer
 */
std::string getIsoDateTime();
/**
 * @brief getNospaceIsoDateTime
 * @return isodate:isotime
 */
std::string getNospaceIsoDateTime();
/**
 * @brief getNospaceIsoDateTime
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
    /// nanoseconds
    long int ns;

    Time()=default;
    /// from nano seconds @param ns
    Time(long int ns):ns(ns){}
    /// from timetype
    Time(long double tt /** @param tt value*/,TIME_TYPE type /** @param type unit of the value */);

    /**
     * @brief toStr
     * @return a human readable string representation
     */
    std::string toStr() const;
    /**
     * @brief toStr
     * @param type output type
     * @return a human readable string representation
     */
    std::string toStr(TIME_TYPE type) const;
    /**
     * @brief operator +=
     * @param rhs
     * @return add the times
     */
    Time& operator+=(const Time& rhs);
    /**
     * @brief operator /=
     * @param rhs
     * @return divide the underlying nanoseconds
     */
    Time& operator/=(const Time& rhs);    

    /**
     * @brief getSeconds
     * @return time in seconds
     */
    double getSeconds();
    /**
     * @brief getMilliSeconds return the time in
     * @return time in milliseconds
     */
    double getMilliSeconds();
    /// set time to value in seconds @param seconds
    void setSeconds(double seconds);

};
/**
 * @brief operator == maps to lsh.ns==rhs.ns
 * @param lhs
 * @param rhs
 * @return
 */
bool operator==(const Time& lhs, const Time& rhs);
/**
 * @brief operator != maps to lsh.ns!=rhs.ns
 * @param lhs
 * @param rhs
 * @return
 */
bool operator!=(const Time& lhs, const Time& rhs);
/**
 * @brief operator < maps to lsh.ns<rhs.ns
 * @param lhs
 * @param rhs
 * @return
 */
bool operator< (const Time& lhs, const Time& rhs);
/**
 * @brief operator > maps to lsh.ns>rhs.ns
 * @param lhs
 * @param rhs
 * @return
 */
bool operator> (const Time& lhs, const Time& rhs);

/**
 * @brief operator <= maps to lsh.ns<=rhs.ns
 * @param lhs
 * @param rhs
 * @return
 */
bool operator<=(const Time& lhs, const Time& rhs);
/**
 * @brief operator >= maps to lsh.ns>=rhs.ns
 * @param lhs
 * @param rhs
 * @return
 */
bool operator>=(const Time& lhs, const Time& rhs);
/**
 * @brief operator + maps to lsh.ns==rhs.ns
 * @param lhs
 * @param rhs
 * @return
 */
Time operator+ (const Time& lhs,const Time& rhs);

/**
 * @brief operator - maps to lsh.ns-rhs.ns
 * @param lhs
 * @param rhs
 * @return
 */
Time operator- (const Time& lhs,const Time& rhs);

/**
 * @brief operator << human readable time representation, will automatically figure out a good time unit
 * @param os
 * @param t
 * @return
 */
std::ostream& operator<<(std::ostream &os, const Time& t);

/**
 * @brief sleep makes this_thread sleep for
 * @param seconds
 */
void sleep(double seconds);
/**
 * @brief sleep makes this_thread sleep for
 * @param milliseconds
 */
void sleep_ms(double milliseconds);
/**
 * @brief sleep_us makes this_thread sleep for
 * @param microseconds
 */
void sleep_us(double microseconds);




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
 * @brief The ScopedDelay struct
 *
 * {ScopedDelay sd(10*1e6);
 * dostuff...
 * } // if dostuff takes less than 10 seconds, we sleep 10-time(dostuff) afterwards
 *
 * So if a loop should run at 10 fps at most, add a ScopedDelay sd(1e5); to the start of it.
 *
 */
struct ScopedDelay{
    std::chrono::time_point<std::chrono::steady_clock,std::chrono::nanoseconds > mark;
    ScopedDelay(double min_delay_us);
    ~ScopedDelay();
};


/**
 * @brief The Timer class
 * High precision Timer
 *
 * Timing is accurate in us to ms range, may be accurate in ns dep on implementation but not likely
 *
 *
 *
 *
 * DISCUSS: Should it be atomic? - No.
 * Thread safety when possible is a good idea,
 *  but multiple threads using one timer is just wrong.
 * The locking takes time which may confuse results with timers in inner loops.
 *
 * Is there a way to ensure the timer isnt called from multiple threads?
 * no no fast way
 *
 */
class Timer{

    /// the last tic
    std::chrono::time_point<std::chrono::steady_clock,std::chrono::nanoseconds > mark;
    /// the last toc
    std::chrono::time_point<std::chrono::steady_clock,std::chrono::nanoseconds > recall;
    /// the time deltas
    std::vector<Time> ts;
    bool dotic=true;
    std::string name="unnamed";


public:
    bool tickable();
    bool tockable();

    Timer();
    Timer(std::string name, uint capacity=1024);
    void reserve(unsigned int initialsize);
    /**
     * @brief tic mark the beginning of a new time delta
     */
    void tic();
    /**
     * @brief toc mark the end of the last time delta. if called prior to any toc behaviour is undefined.
     * @return Time in nanoseconds
     */
    Time toc();


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


    /**
     * @brief toStr
     * @return human readable timer information
     */
    std::string toStr() const;
    std::vector<std::string> toStrRow() const;

    /**
     * @brief clear clears the vector with time deltas
     */
    void clear();
    /**
     * @brief getTimes
     * @return the time deltas stored so far
     */
    std::vector<Time> getTimes();       // alias for Time
    uint64_t samples(){return ts.size();}
    /**
     * @brief getSum
     * @return the sum
     */
    Time getSum() const;
    /**
     * @brief getMedian
     * @return the median
     */
    Time getMedian() const;
    /**
     * @brief getMean
     * @return the mean
     */
    Time getMean() const;
    /**
     * @brief getMax
     * @return the max
     */
    Time getMax() const;
    /**
     * @brief getMin
     * @return the min
     */
    Time getMin() const;



};




class NamedTimerPack{
public:
    std::map<std::string, Timer> ts;
    Timer& make_or_get(std::string name);
    void tic(std::string name);
    void toc(std::string name);
};


std::ostream& operator<<(std::ostream &os, NamedTimerPack ntp);

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



#endif // Time_H
