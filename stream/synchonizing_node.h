#pragma once

#include <mlib/stream/node.h>
#include <mlib/utils/cvl/sync_priority_queue.h>

namespace cvl{




template<class Element, class time_seconds>

/**
 * @brief The SequentializerNode class
 *
 * Reorders measurements based on their time
 *
 * accumulates delay_s of measurements,
 *  then outputs them in order,
 *  measurements which arrive later than that are discarded.
 *
 */
class SequentializerNode:public Node<Source<Element>,Sink<Element>>
{
    std::atomic<double> t_latest_added{std::numeric_limits<double>::lowest()};
    int thrown_away=0;
    double delay_s;
    std::mutex tsbn_mtx;
    std::condition_variable tsbn_cv;
    // lower is higher priority
    PriorityQueue<Element,time_seconds> priority_queue;

public:
    using NT = Node<Source<Element>,Sink<Element>>;
    virtual bool process(Element& input, Element& output) override
    {
        output=input;
        return true;
    };

    // override for things like a max size
    virtual void sink_(Element& input) override {
        std::unique_lock ul(tsbn_mtx);
        double t1=time_seconds(input);
        // throw away anything that was too disordered to be synchronized
        if(t1<t_latest_added){
            thrown_away++;
            return;
        }
        priority_queue.push(input);
        for(auto it=priority_queue.begin();it!=priority_queue.end();++it){
            // they are ordered, so if the one I am at is too late, stop
            if(it->first>=t1-delay_s) break;
            NT::input_queue.push(it->second);
            t_latest_added=it->first;
            priority_queue.erase(it);
        }
    }
};



}


