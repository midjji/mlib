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

namespace cvl {


template<class Input_>
class Sink{
public:
    using Input=Input_;

    // this function should always return fast.
    // this function is always thread safe!
    // really need decorators for that...
    void sink(Input& input){        if(!slumbering)            sink_(input);}
    void set_slumbering(bool slumber){        slumbering=slumber;    }

    virtual ~Sink(){}
protected:
    virtual void init(){}
    virtual void sink_(Input& input)=0;
    std::atomic<bool> slumbering{false};
};

/**
 * @brief The NoSink struct
 * If a node has no inputs, use this one!
 */
struct NoSink: public Sink<int>{
    using Input=typename Sink::Input;
    virtual ~NoSink(){}
private:
    void sink_([[maybe_unused]] Input& input) override{}
};
}
