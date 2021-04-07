#pragma once

#include <mlib/stream/node.h>


namespace cvl{

template<class Element>
/**
 * @brief The BufferNode class
 * \brief Caches up to max_size elements, blocks with max_size is reached
 */
class BufferNode:public Node<Source<Element>,Sink<Element>> {
uint max_size;
public:

    BufferNode(uint max_size=std::numeric_limits<uint>::max()):max_size(max_size){}

    virtual bool process(Element& input, Element& output){output=input;return true;};

    // override for things like a max size
    virtual void sink_(Element& input) override{
        auto stop=[&](){return !Node<Source<Element>,Sink<Element>>::running;};
        Node<Source<Element>,Sink<Element>>::input_queue.blocking_push(input,stop,max_size);
    }
};



template<class Element>
class RAMLimitedBufferNode:public Node<Source<Element>,Sink<Element>> {

    std::atomic<double> storage_mb{0};
std::atomic<double> storage_limit_mb;
std::mutex tsbn_mtx;
std::condition_variable tsbn_cv;
public:
using NT = Node<Source<Element>,Sink<Element>>;

    RAMLimitedBufferNode(double storage_limit_mb=100):storage_limit_mb(storage_limit_mb){}

    virtual bool process(Element& input, Element& output) override
    {
        std::unique_lock ul(tsbn_mtx);
        storage_mb=storage_mb - input->storage_megabytes();
        output=input;
        ul.unlock();
        tsbn_cv.notify_one();
        return true;
    };

    // override for things like a max size
    virtual void sink_(Element& input) override {
        std::unique_lock ul(tsbn_mtx);
        tsbn_cv.wait(ul, [&](){
            return (storage_mb<storage_limit_mb)||
                    NT::input_queue.size()<1 ||
                    !NT::running;});
        if(!NT::running) return;
        NT::input_queue.push(input);
        storage_mb=storage_mb+input->storage_megabytes();
        ul.unlock();
        tsbn_cv.notify_one();
    }
};


template<class Element, class time_seconds>
/**
 * @brief The TimeBufferNode class
 * \brief A time buffer sink blockingly waits
 * for current elements time - last popped element time is less than span,
 * or the queue is empty.
 *
 * In other words it caches up to span seconds ahead.
 * It may reorder the stream if the elements are not sorted by time
 *
 */
class TimeSpanBufferNode:public Node<Source<Element>,Sink<Element>> {
std::atomic<double> t0;
double span_s;
std::mutex tsbn_mtx;
std::condition_variable tsbn_cv;
public:
using NT = Node<Source<Element>,Sink<Element>>;

    TimeSpanBufferNode(double span_s):span_s(span_s){}

    virtual bool process(Element& input, Element& output) override
    {
        std::unique_lock ul(tsbn_mtx);
        double t=time_seconds(input);
        if(t0<t)t0=t;

        output=input;
        ul.unlock();
        tsbn_cv.notify_one();
        return true;
    };

    // override for things like a max size
    virtual void sink_(Element& input) override {
        std::unique_lock ul(tsbn_mtx);
        double t1=time_seconds(input);
        tsbn_cv.wait(ul, [&](){
            return (t1 < t0+span_s)||
                    NT::input_queue.size()<1 ||
                    !NT::running;});
        NT::input_queue.push(input);
    }

};



}


