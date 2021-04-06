#pragma once
/********************************** FILE ************************************/
/** \file    source.h
 *
 * \brief a data source, used with mlib/stream/node.h
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

#include <memory>
#include <mutex>
#include <vector>
#include <mlib/stream/sink.h>

namespace cvl {
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
                q->sink(output);
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





}
