#if 0
#pragma once

#include <vector>
#include <thread>
#include <iostream>

#include <assert.h>

class StreamData{
public:
    virtual ~StreamData();
};

class StreamOperator;

class Stream{

public:
    enum class STATE{BREAK,CONTINUE,NORMAL};
    static std::string stream_state_2_string(Stream::STATE state){
        // enums allow case, but are just sequential numbers so you cant overload << without messing up <<1
        if(state==Stream::STATE::BREAK) return "Break"; // end of stream
        if(state==Stream::STATE::CONTINUE) return "Continue"; // means skip this data
        if(state==Stream::STATE::NORMAL) return "Normal"; // continue as usual
        return "Undefined!";
    }

    Stream(){}
    ~Stream(){done=true;thr.join();}
    void addOperator(std::shared_ptr<StreamOperator> so);
    void start(){       thr=std::thread(&Stream::run,std::ref(*this));}
    void stop(){done=true;}
private:
    std::thread thr;
    bool done=false;
    void run();
    std::vector<std::shared_ptr<StreamOperator>> streamops;

};


class StreamOperator{
public:
    virtual ~StreamOperator();
    virtual Stream::STATE operator()([[maybe_unused]] std::shared_ptr<StreamData>& sd){return Stream::STATE::BREAK;}
    virtual std::string name()=0;
    virtual void stop(){}
};

#endif
