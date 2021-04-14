#pragma once
/** \file    node.h
 *
 * \brief The node class represents the nodes of a thread safe asynchronous processing stream/graph.
 *
 * \desc
 * The standard node processes its inputs using node->process(input) whenever input is available.
 * If this generates output, that is pushed to each listening node.
 *
 *
 * Any number of threads may push to a node by
 * node->input_queue()->push() // which always returns instantly, but throws away data if the queue is full
 * or
 * node->input_queue()->blocking_push() // which hangs if the queue is full.
 *
 * Node has a single input, but multiple threads can write to this input.
 * Node has a single output, but multiple listeners can listen to this output.
 * MIMONode has multiple inputs and multiple outputs.
 *
 * you add a listener by
 * node->add_queue()
 * or node->add_queue(queue)
 *
 * You can explicitly set which queue is used as input,
 * if e.g you have something that has an output queue, then just grab that and give as input.
 * either on creation: node(input)
 * or using
 * node->set_input_queue(input)
 *
 * The node can be set to
 * slumbering which means all inputs are discarded instead of processed.
 * This is useful for setup.
 *
 *
 * The node owns its own input queue, but does not own its listeners.
 *
 * \discussion
 * This stream system is designed for when each node should perform heavy processing and its inputs are typically smartptrs to big blocks of data.
 * The point of using the class is to avoid common threading errors.
 * The point of templating the type is to give compiler verified streams.
 * If you set T to some base class which is then derived, this guarantee goes away.
 * Yes its possible to make it faster.
 * Yes it can be more generic.
 * This is for the specific case where it is good.
 *
 *
 *
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






#include <vector>
#include <thread>
#include <memory>
#include <mlib/utils/cvl/syncque.h>
#include <mlib/utils/mlog/log.h>
#include <mlib/utils/mlibtime.h>
#include <mlib/stream/sink.h>
#include <mlib/stream/source.h>


namespace cvl{




template<class Source=NoSource, class Sink=NoSink>
/**
 * @brief The Node class, see top of file!
 *
 * Defaults to a thread which only runs process, but using the Sink and Source
 * instead you get input and output...
 *
 * Node is single input from multiple sources, and single output to multiple sinks.
 *
 *
 */
class Node :  public Source, public Sink
{
public:
    using Input=typename Sink::Input;
    using Output= typename Source::Output;


    virtual ~Node() {        
        running=false;
        input_queue.notify_all();
        if(node_thr.joinable())
            node_thr.join();
    }
    // must be templated on the node type,
    // since static does not know
    template<class NodeType, class... Args> static std::shared_ptr<NodeType>
    create(Args... args) {
        std::shared_ptr<NodeType> ipp(new NodeType(args...));
        ipp->wp_self=ipp; // enable_shared_from_this is bugged in c++17


        // the node should not own itself. hence by reference, [&]
        ipp->node_thr=std::thread([&](){
            mlog().set_thread_name(ipp->node_name());
            ipp->running=true;
            std::unique_lock<std::mutex> ul(ipp->start_mutex);
            ipp->start_cv.wait(ul, [&](){return ipp->ready || !ipp->running;});
            ipp->init();
            ipp->loop();
            ipp->running=false;
        });
        return ipp;
    }

    std::shared_ptr<Node> get_node(){wp_self.lock();}
    std::shared_ptr<Sink> get_sink(){return std::static_pointer_cast<Sink>(wp_self.lock());}
    std::shared_ptr<Source> get_source(){return std::static_pointer_cast<Source>(wp_self.lock());}

    // by default the node does not process input untill you tell it to start
    // this is to make sure you can setup all listeners first.
    void start(){
        //must use a lock despite the variable beeing atomic for the cv to work properly, its a bug in the standard
        std::unique_lock<std::mutex> ul(start_mutex);
        ready=true;
        ul.unlock();
        start_cv.notify_one();
    }

protected:
    std::mutex start_mutex;
    std::condition_variable start_cv;
    Node(){}
    //override process to do stuff
    virtual bool process([[maybe_unused]] Input& input,
    [[maybe_unused]] Output& output){return false;};

    // override for things like a max size
    virtual void sink_(Input& input) override{
        input_queue.push(input);
    }

    virtual std::string node_name(){ return "Node";}
    virtual void init() override{
        this->Sink::init();
        this->Source::init();
    };

    /**
     * @brief loop
     * the caller must exist for the duration of this loop,
     * which is guaranteed by the thr.join in  Node
     */
    virtual void loop()
    {
        while(running) {
            Input in;
            auto stop=[&](){return !running;};
            if(!input_queue.blocking_pop(in,stop)) break;
            Output out;
            if(!process(in, out)) continue;// may have sideeffects
            this->push_output(out);
        }
    }
    std::atomic<bool> running{false};
    std::atomic<bool> ready{false};
    SyncQue<Input> input_queue;
private:

    std::thread node_thr;
    std::weak_ptr<Node> wp_self;
};



}

