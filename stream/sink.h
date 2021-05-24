#pragma once
/********************************** FILE ************************************/
/** \file    sink.h
 *
 * \brief a data sink, used with mlib/stream/node.h
 *
 * \remark
 * - c++11
 * - no dependencies
 * - header only
 *

 *
 * \todo
 *
 *
 *
 *
 * \author   Mikael Persson
 * \date     2010
 * \note
 *
 ******************************************************************************/
#include <atomic>
#include <chrono>
#include <mlib/utils/cvl/syncque.h>

namespace cvl {


template<class Input_>
class Sink{
public:
    using Input=Input_;

    // this function should always return fast.
    // it claims ownership of input, so copies if needed
    void sink(Input input){        if(slumbering) return;        sink_(input);    }
    void set_slumbering(bool slumber){        slumbering=slumber;    }
    bool is_slumbering(){return slumbering;}
    virtual ~Sink(){}
protected:
    virtual void init(){}
    virtual void sink_(Input& input)=0;
    std::atomic<bool> slumbering{false};
private:

};

/**
 * @brief The NoSink struct
 * If a node has no inputs, use this one!
 */
struct NoSink: public Sink<int>{
    using Input=typename Sink<int>::Input;
    virtual ~NoSink(){}
private:
    void sink_([[maybe_unused]] Input& input) override{}
};


template<class T>
struct QueueSink:public Sink<T> {
    using Input=T;
  void sink_(Input& input) override{
      queue.push(input);
  }
  SyncQue<T> queue;
};
}
