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
 * - Should we count the number of elements separately from queue.empty() in an atomic int?
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
 * a thread safe deque, pop throws EOF exception
 * if the syncque is stopped ( allows for exitable thread blocking)
 *
 * Note that in most cases the elements should be pointers,
 * but in the case of eg ints, they shouldnt be to maximize performance.
 * Avoid recursive locks, its slow and always avoidable
 */
template <class T>
class SyncQue{

private:
    std::atomic<bool> stopped;
    std::atomic<bool> slumbering;
    std::atomic<uint32_t> max_size;
    std::deque<T> que;
    std::mutex mtx;
    std::condition_variable cv;


    //    void resize_toss_front(){
    //        SyncQue<T> tmp;
    //        ...
    //#error "implement!"
    //    }

public:

    explicit SyncQue( uint32_t max_size_ = std::numeric_limits<uint32_t>::max()) : stopped(false), slumbering(false),max_size(max_size_) {
        if(max_size==0)
            max_size=std::numeric_limits<uint32_t>::max();
    }
    SyncQue& operator = (SyncQue&) = delete;


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
    bool is_stopped(){return stopped;}

    void set_max_size(int size){
        std::unique_lock<std::mutex> ul(mtx); // locks, unlocked as it goes out of scope
        max_size = size;
        while (max_size > 0 && que.size() >= max_size)
            que.pop_front(); // might be made faster?
        ul.unlock();
        cv.notify_one();
    }

    void push(const T& t){
        //std::cout<<"name: "<<name <<" push "<<n++<<std::endl;
        std::unique_lock<std::mutex> ul(mtx); // locks, unlocked as it goes out of scope
        if(stopped) return;
        if(slumbering) return;

        while (que.size() >= max_size)
            que.pop_front(); // should only happen once here

        que.push_back(t);
        ul.unlock();
        cv.notify_one();
        //std::cout<<"push done"<<std::endl;
    }


    // blocking, only fails if the que is stopped
    bool blocking_try_pop(T& t){
        //  std::cout<<"name: "<<name <<" pop "<<k++<<std::endl;
        std::unique_lock<std::mutex> ul(mtx); // locks
        cv.wait(ul, [this](){return !que.empty() || stopped;});


        if(stopped){
            std::cout<<"pop aborted"<<std::endl;
            ul.unlock();
            cv.notify_all();
            return false;
        }

        t = que.front();
        que.pop_front();
        ul.unlock();
        cv.notify_one();
        // std::cout<<"pop done"<<std::endl;
        return true;
    }
    /// blocking, throws eof if stopped, should only be used if internal messages can stop things...
    /*
    T pop(){
      //  std::cout<<"name: "<<name <<" pop "<<k++<<std::endl;
        std::unique_lock<std::mutex> ul(mtx); // locks
        cv.wait(ul, [this](){return !que.empty() || stopped;});

        if(stopped){
            std::cout<<"pop aborted"<<std::endl;
            ul.unlock();
            cv.notify_all();
            throw std::runtime_error("Syncque stopped: ");
        }

        T t = que.front();
        que.pop_front();
        ul.unlock();
        cv.notify_one();
       // std::cout<<"pop done"<<std::endl;
        return t;
    }*/


    T pop_back_and_clear(){
        std::unique_lock<std::mutex> ul(mtx); // locks
        cv.wait(ul, [this](){return !que.empty() || stopped;});

        if(stopped){
            std::cout << "pop aborted" << std::endl;
            ul.unlock();
            cv.notify_all();
            throw std::runtime_error("Syncque stopped: ");
        }

        T t = que.back();
        que.clear();
        ul.unlock();
        cv.notify_one();
        return t;
    }

    T peek(){
        std::unique_lock<std::mutex> ul(mtx); // locks
        cv.wait(ul,[this](){return !que.empty() || stopped;});
        if(stopped){
            std::cout << "pop aborted" << std::endl;
            ul.unlock();
            cv.notify_all();
            throw new std::runtime_error("Syncque: End of file error: ");
        }
        T t=que.front();
        return t;
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


    uint size(){
        std::unique_lock<std::mutex> ul(mtx); // locks
        return uint(que.size());
    }

    void stop(){
        //std::cout<<"stopping"<<std::endl;
        std::unique_lock<std::mutex> ul(mtx); // locks
        stopped=true;
        ul.unlock();
        cv.notify_all();
    }
    void clear(){
        std::unique_lock<std::mutex> ul(mtx); // locks
        que.clear();
        ul.unlock();
        cv.notify_all();
    }
    void waitForEmpty(){
        std::unique_lock<std::mutex> ul(mtx); // locks
        cv.wait(ul,[this](){return que.empty() || stopped;});

    }
    void wait_for_size_less_than(uint size){
        std::unique_lock<std::mutex> ul(mtx); // locks   
        cv.wait(ul,[&](){return (que.size()<size) || stopped;});
    }
    void wait_on_size_push(){
        std::unique_lock<std::mutex> ul(mtx); // locks
        cv.wait(ul,[&](){return (que.size()<max_size-1) || stopped;});
    }


private:
    int name;
};




}// end namespace cvl
#endif // SYNCQUE_H
