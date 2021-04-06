#pragma once
/********************************** FILE ************************************/
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





namespace cvl{

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



template<class Output_>
class Source {    
public:    
    using Output=Output_;
    virtual ~Source(){}
    virtual void add_sink(std::shared_ptr<Sink<Output>> queue) {
        if(queue==nullptr) return;
        std::unique_lock<std::mutex> ul(node_mtx);
        queues.reserve(100);
        queues.push_back(queue);
    }

protected:
    virtual void init(){}
    void push_output(Output& output)    {
        std::unique_lock<std::mutex> ul(node_mtx);
        int missing=0;
        for(auto& wq:queues) {
            auto q=wq.lock();
            if(q) // !=nullptr
                q->push(output);
            else
                missing++;
        }
        if(missing>10 && missing*2 >queues.size())
            clear_missing();
    }
private:
    void clear_missing()    {
            auto tmp=queues;
            queues.clear();
            for(auto& wp:tmp)
                if(wp.lock())
                    queues.push_back(wp);
    }
    // changes are rare, insert cheap, removal expensive
    std::vector<std::weak_ptr<Sink<Output>>> queues;
    std::mutex node_mtx;
};

struct NoSource:public Source<int>{
using Output= typename Source::Output;
    virtual ~NoSource(){}
protected:
    void push_output([[maybe_unused]] Output& output){}
    void init() override{}
private:
    void add_sink([[maybe_unused]] std::shared_ptr<Sink<Output>> queue) override {}
};




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
class Node : public Sink, public Source
{
public:
    using Input=typename Sink::Input;
    using Output= typename Source::Output;

    Node(){}
    virtual ~Node() {
        running=false;
        if(node_thr.joinable())
            node_thr.join();
    }
    // must be templated on the node type,
    // since static does not know
    template<class NodeType, class... Args> static std::shared_ptr<NodeType>
    start(Args... args) {
        std::shared_ptr<NodeType> ipp(new NodeType(args...));
        ipp->init();
        ipp->self=ipp; // enable_shared_from_this is bugged in c++17
        // the node should not own itself. hence by reference, [&]
        ipp->node_thr=std::thread([&](){
            ipp->loop();
        });
        return ipp;
    }

    std::shared_ptr<Node> get_node(){return wp_self.lock();}
    std::shared_ptr<Sink> get_sink(){return wp_self.lock();}
    std::shared_ptr<Source> get_source(){return wp_self.lock();}

protected:
    virtual std::string node_name(){ return "Node";}
    virtual void init(){
        this->Sink::init();
        this->Source::init();
        running=true;
    };
    virtual bool process(Input& input,
                         Output& output) =0;

    /**
     * @brief loop
     * the caller must exist for the duration of this loop,
     * which is guaranteed by the thr.join in  Node
     */
    virtual void loop()
    {
        mlog().set_thread_name(node_name());
        while(running) {
            Input in;
            auto stop=[&](){return !running;};
            if(!this->input->blocking_pop(in,stop)) break;
            Output out;
            if(!process(in, out)) continue;// may have sideeffects
            this->push_output(out);
        }
        running=false;
    }

private:
    std::atomic<bool> running;
    std::thread node_thr;
    std::weak_ptr<Node> wp_self;
};



}

