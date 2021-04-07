#pragma once
/* ********************************* FILE ************************************/
/** \file    sync_priority_queue.h
 *
 * \brief    This header contains a syncronized thread safe priority queue
 *
 *
 * \remark
 * - c++11
 * - self contained(just .h
 * - no dependencies
 * - os independent works in linux, windows etc are untested for some time
 *
 *
 *
 *
 *
 * \todo
 *
 *
 *
 * \author   Mikael Persson
 * \date     2014-07-01
 * \note BSD licence
 *
 *
 ******************************************************************************/



#include <map>
#include <mutex>
#include <thread>
#include <iostream>
#include <condition_variable>
#include <exception>
#include "assert.h"

#include <iostream>
#include <atomic>
#include <limits>

typedef unsigned int uint;
// sync que
namespace cvl{


template <class Value, // the type
          class KeyofValue, // gets the key from the value
          class Key=typename std::invoke_result<KeyofValue>::type , // the key type
          class Less=std::less<Key>>
/**
 * @brief The PriorityQue class
 * lower is higher priority, this is unsynchronized, but used in the syncpriorityqueue
 *
 * why the hell is this not a default structure in c++11?
 */
class PriorityQueue {

private:
    // can have multiple values with the same key,
    std::multimap<Key,Value,Less> map;
    std::mutex mtx;
    std::condition_variable cv;
public:

    void push(const Value& t) {
        map.insert(KeyofValue(t),t);
    }

    bool try_pop(Value& t){
        if(map.size() > 0){
            t=map.begin()->second;
            map.erase(map.begin());
            return true;
        }
        return false;
    }

    uint size(){
        return uint(map.size());
    }

    void clear(){
        map.clear();
    }
    auto begin(){return map.begin();}
    auto end(){return map.end();}
    template<class T> void erase(T t){        map.erase(t);    }
};






template <class Value, // the type
          class KeyofValue, // gets the key from the value
          class Key=typename std::invoke_result<KeyofValue>::type , // the key type
          class Less=std::less<Key>>
/**
 * @brief The SyncPriorityQue class
 * lower is higher priority
 */
class SyncPriorityQueue {

private:
    // can have multiple values with the same key,
    std::multimap<Key,Value,Less> map;
    std::mutex mtx;
    std::condition_variable cv;
public:

    SyncPriorityQueue& operator = (SyncPriorityQueue&) = delete;
    std::shared_ptr<SyncPriorityQueue> create(){
        return std::make_shared<SyncPriorityQueue>();
    }
    void push(const Value& t)
    {
        std::unique_lock<std::mutex> ul(mtx); // locks, unlocked as it goes out of scope
        map.insert(KeyofValue(t),t);
        ul.unlock();
        cv.notify_one();
        //std::cout<<"push done"<<std::endl;
    }

    bool try_pop(Value& t){
        std::unique_lock<std::mutex> ul(mtx); // locks
        if(map.size() > 0){
            t=map.begin()->second;
            map.erase(map.begin());
            return true;
        }
        return false;
    }

    template<class StopCondition>// [&running](){return !running; /*running must be atomic!*/}
    bool blocking_pop(Value& t,
                          StopCondition stop){
        //  std::cout<<"name: "<<name <<" pop "<<k++<<std::endl;
        std::unique_lock<std::mutex> ul(mtx); // locks
        cv.wait(ul, [&](){return !map.empty() || stop();});
        if(map.size() > 0){
            t=map.begin()->second;
            map.erase(map.begin());
            ul.unlock();
            cv.notify_all();
            return true;
        }
        ul.unlock();
        cv.notify_all();
        // std::cout<<"pop done"<<std::endl;
        return false;
    }

    uint size(){
        std::unique_lock<std::mutex> ul(mtx); // locks
        return uint(map.size());
    }

    void clear(){
        std::unique_lock<std::mutex> ul(mtx); // locks
        map.clear();
        ul.unlock();
        cv.notify_all();
    }


    template<class StopCondition>// [&running](){return !running; /*running must be atomic!*/}
    void waitForEmpty(StopCondition stop){
        std::unique_lock<std::mutex> ul(mtx); // locks
        cv.wait(ul,[&](){return map.empty() || stop();});
        ul.unlock();
        cv.notify_all();
    }
};




}// end namespace cvl


