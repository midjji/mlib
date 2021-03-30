#pragma once
/* ********************************* FILE ************************************/
/** \file    synque.h
 *
 * \brief    This header contains a syncronized thread safe que
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

#ifndef SYNCQUE_H
#define SYNCQUE_H

#include <deque>
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



/**
 * @brief The SyncQue<T> class
 * a thread safe deque
 *
 * Note that in most cases the elements should be pointers,
 * but in the case of eg ints, they shouldnt be to maximize performance.
 *
 * Avoid recursive locks, its slow and always avoidable
 */
template <class T>
class SyncQue{

private:    
    std::deque<T> que;
    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<bool> slumbering{false};

public:    

    SyncQue& operator = (SyncQue&) = delete;
    std::shared_ptr<SyncQue<T>> create(){
        return std::make_shared<SyncQue<T>>();
    }
    void set_slumbering(bool slumber_val, bool clear_queue=true){
        std::unique_lock<std::mutex> ul(mtx); // locks, unlocked as it goes out of scope
        slumbering = slumber_val;
        if (clear_queue) que.clear();
        ul.unlock();
        cv.notify_one();
    }
    bool is_slumbering(){
        //std::unique_lock<std::mutex> ul(mtx); // locks, unlocked as it goes out of scope
        return slumbering;
    }

    void push(const T& t){
        //std::cout<<"name: "<<name <<" push "<<n++<<std::endl;
        std::unique_lock<std::mutex> ul(mtx); // locks, unlocked as it goes out of scope
        if(slumbering) return;
        que.push_back(t);
        ul.unlock();
        cv.notify_one();
        //std::cout<<"push done"<<std::endl;
    }
    template<class StopCondition>// [&running](){return !running; /*running must be atomic!*/}
    void blocking_push(const T& t,
                       StopCondition stop,
                       uint max_size=std::numeric_limits<uint>::max()){
        std::unique_lock<std::mutex> ul(mtx); // locks, unlocked as it goes out of scope
        cv.wait(ul, [&](){return que.size()<max_size || stop();});
        que.push_back(t);
        ul.unlock();
        cv.notify_one();
    }

    bool try_pop(T& t){
        std::unique_lock<std::mutex> ul(mtx); // locks
        if(que.size() > 0){
            t = que.front();
            que.pop_front();
            return true;
        }
        return false;
    }
    template<class StopCondition>// [&running](){return !running; /*running must be atomic!*/}
    bool blocking_pop(T& t,
                          StopCondition stop){
        //  std::cout<<"name: "<<name <<" pop "<<k++<<std::endl;
        std::unique_lock<std::mutex> ul(mtx); // locks
        cv.wait(ul, [&](){return !que.empty() || stop();});
        if(!que.empty())
        {
            t = que.front();
            que.pop_front();
        }
        ul.unlock();
        cv.notify_all();
        // std::cout<<"pop done"<<std::endl;
        return true;
    }

    uint size(){
        std::unique_lock<std::mutex> ul(mtx); // locks
        return uint(que.size());
    }

    void clear(){
        std::unique_lock<std::mutex> ul(mtx); // locks
        que.clear();
        ul.unlock();
        cv.notify_all();
    }


    template<class StopCondition>// [&running](){return !running; /*running must be atomic!*/}
    void waitForEmpty(StopCondition stop){
        std::unique_lock<std::mutex> ul(mtx); // locks
        cv.wait(ul,[&](){return que.empty() || stop();});
        ul.unlock();
        cv.notify_all();
    }
};




}// end namespace cvl
#endif // SYNCQUE_H
