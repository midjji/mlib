#pragma once
#include <mlib/utils/cvl/syncque.h>
#include <mlib/utils/mlibtime.h>
namespace cvl{

template<class Stream>
class BufferedStream
{
public:
    using sample_type=typename Stream::sample_type;

    template<class... Args>
    BufferedStream(uint offset,
                   std::shared_ptr<Stream> ds):
        offset(offset),ds(ds){
        running=true;
        thread=std::thread([&] { loop(); running=false;});
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
        auto stop=[&](){return !running;};
        queue.blocking_pop(sd, stop); // return value ignored...
        return sd; // may return nullptr if stream stopped!
    }
    int samples() const {return ds->samples();}


    uint offset=0;
    std::shared_ptr<Stream> ds;

private:


    std::atomic<bool> running;
    std::thread thread;
    SyncQue<sample_type> queue;
    mlib::Timer loadtimer=mlib::Timer("load timer");


    void loop(){
        //std::cout<<"looping"<<std::endl;
        int samples=ds->samples();
        for(uint index=offset;
            running && (int(index)<samples);
            ++index)
        {
            while(queue.size()>5 && running)    mlib::sleep_ms(10);
            if(!running)
                break;
            loadtimer.tic();
            auto sample=ds->sample(index);

            loadtimer.toc();            
            queue.push(sample);
        }
        //std::cout<<"looping done"<<std::endl;
    }
};
}// end namespace cvl

