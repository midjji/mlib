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
private:
     int max_size_;

public:
    std::vector<T> buff;
    std::vector<T> buffert() const{return buff;}
    const T& operator[](int index) const{return buff[index];}
    int max_size() const{return max_size;};
    int size() const {return buff.size();}

    LimitedHistory(int max_size_=5):max_size_(max_size_){}



    void push(T t){
        if(int(buff.size())<max_size_)
            buff.push_back(t);
        else{
            // slow as hell but very easy to get right!
            for(unsigned int i=0;i<buff.size()-1;++i)
                buff[i]=buff[i+1];
            buff[buff.size()-1]=t;
        }
    }
    bool in(T t){
        for(const T& v:buff)
            if(t==v) return true;
        return false;
    }
    auto begin(){return buff.begin();}
    auto end(){return buff.end();}
};
