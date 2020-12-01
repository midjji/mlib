#pragma once
#include <vector>
template<class T>
/**
 * @brief The LimitedHistory struct
 * //TODO make fast,
 *
 * its not a fifo, it wraps it, its not a circ buffer, but wraps it,
 * I keep needing this.
 */
struct LimitedHistory{
    unsigned int size;
    LimitedHistory(unsigned int size=5):size(size){}
    std::vector<T> buff;


    void push(T t){
        if(buff.size()<size)
            buff.push_back(t);
        else{
            // slow as hell but very easy to get right!
            for(unsigned int i=0;i<buff.size()-1;++i)
                buff[i]=buff[i+1];
            buff[buff.size()-1]=t;
        }
    }
    bool in(T t){
        for(T v:buff)
            if(t==v) return true;
        return false;
    }
    auto begin(){return buff.begin();}
    auto end(){return buff.end();}
};
