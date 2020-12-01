#pragma once
#include <mlib/utils/cvl/syncque.h>
namespace cvl{

template<class Dataset>
class BufferedStream
{
public:
    using sample_type=typename Dataset::sample_type;

    template<class... Args>
    BufferedStream(uint offset,
                   Args... args):
        offset(offset),ds(std::make_shared<Dataset>(args...)){
        running=true;
        thread=std::thread([&] { loop(); });
    }
    ~BufferedStream(){

        running=false;
        queue.clear();
        if(thread.joinable())
            thread.join();
    }
    sample_type next()
    {
        sample_type sd=nullptr;
        queue.blocking_try_pop(sd); // return value ignored...
        return sd; // may return nullptr if stream stopped!
    }


    uint offset=0;
    std::shared_ptr<Dataset> ds;

private:


    std::atomic<bool> running;
    std::thread thread;
    SyncQue<sample_type> queue;
    mlib::Timer loadtimer=mlib::Timer("load timer");


    void loop(){
        //std::cout<<"looping"<<std::endl;
        for(uint index=offset;
            running && index<ds->samples();
            ++index)
        {
            queue.wait_for_size_less_than(5);
            if(!running)
                break;
            loadtimer.tic();
            auto sample=ds->get_sample(index);
            loadtimer.toc();
            queue.push(sample);
        }
        //std::cout<<"looping done"<<std::endl;
    }
};
}// end namespace cvl
