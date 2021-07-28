
#pragma once
/* ********************************* FILE ************************************/
/** \file    mgenerator.hpp
 *
 * \brief    This header contains generator examples
 *
 *
 * \remark
 * - c++17
 * - no dependencies
 * - header only
*/

#include <sstream>
#include <iostream>
#include <tuple>
#include <vector>
#include <functional>
#include <assert.h>
#include <type_traits>
#include <iostream>

struct IndexableGenerator
{
    std::vector<double> ts;
    double t0;
    double delta;
    int max_index;
    IndexableGenerator(std::vector<double> ts):ts(ts), max_index(ts.size()){}
    IndexableGenerator(double t0, double delta=1, int max_index=std::numeric_limits<int>::max()):t0(t0),delta(delta),max_index(max_index){}
    double operator[](int index){
        if(ts.empty()) return t0+ index*delta;
        return ts[index % max_index];
    }
};


#if 0
using namespace std;
namespace cvl {


template<class T=int> struct Range{ // this is an iterator!
    T r0; // the first value
    T delta;// the difference
    T r1; // upper bound, note that for floating point types it is [r0,r1) ie less than r1!
    mutable uint64_t index; // the current index!
    uint64_t size; // the number of elements!


    Range(T r0, T delta=T(1), T r1=std::numeric_limits<T>::max()):r0(r0),delta(delta),r1(r1), size((r1-r0)/delta){}
    // forward iterable only, iterator is immedeatly invalidated on
    const T* begin()     const{}
    const T* end()       const{ return nullptr;}
    const T* cbegin()    const{return begin();}
    const T* cend()      const{ return end();}
};



class positive_numbers_mod_uintmax{
public:
    positive_numbers_mod_uintmax(int index=0):index(index){}
    positive_numbers_mod_uintmax begin(){ return positive_numbers_mod_uintmax(0);    }
    positive_numbers_mod_uintmax end()  { return positive_numbers_mod_uintmax(std::numeric_limits<uint>::max());    }

    // prefix, inc first, then return
    positive_numbers_mod_uintmax operator++()    {         index++;return *this;  }
    // post fix, return value before increment, does for each need this?
    //positive_numbers_mod_uintmax operator++(int) { }
    uint operator*() const{        return index;    }
    bool operator != (positive_numbers_mod_uintmax other){return index!=other.index;}
private:
    uint index=0;
};

}

/**
  * What kind of ranges do I want to have?
  * Well I want for(int i: range(0,2,10)), and for(int i: range(10,-1,0))
  * possibly in python format,
  * or well, do I want this? isnt this just used for shitty views?
  * kinda, but not always. consider plotting a function over a range of values.
  * it could be a view, but that would be very complicated.
  * as well as for(float f: range(0,2.5,10))
  * also range().back(), and range().size(), and range().reverse()
  * I want it to deal with the float problems in a good way, but that will make it slow.
  * iterating over an int is trivial, reverse iterate is simple but easy to get wrong, this is the class you use to
  * not need to make mistakes when speed does not matter.
  * the first kind is the fast and exact integer range.
  * the second kind is the slow but consistent float range.
  * the third kind is the fast forward only consistent float range.
  *
  * I rarely use the latter two. But there is a ton of special cases
  *  that should be dealt with but a dedicated construct.
  *
  * The first special case is wraparound.
  * The second is nans
  *
  * It would be easier if this did not need to be fast, but if it isnt, then I would make the errors in fast code.
  * The simple implementation is, well slower, a multiply every time, both ints and floats, but its easy to write.
  *
  * Note the standard simple first argument does not hold, because this is supposed to be a generally useful lib.
  * However, perhaps I should write it the slow way first, and see how much I use it,
  * also perhaps the compiler is clever?
  *
  * Float ranges are really hard,
  * basically repeating current +=delta N times is different from start + N*delta.
  *
  * Its very unfortunate that != is used, instead of < or even "in"!
  *
  *
  *
  * ok, so speed wise I wont use these in anything critical.
  * non speed wise there are two cases
  *
  * range(start, stop, delta)
  * range(start, stop, steps) # problematically similar
  * if delta>0 its
  * if start< stop, start+index*delta
  * if start>stop, start-index*delta
  * so delta>0 gives fewer special cases,
  * no matter what its a
  *
  * integer ranges are quite different from float ranges.
  * unsigned vs signed is a problem too...
  *
  * practially every line is subtly different,
  * the only one which is straightforward is the double one.
  * nope the double one has issues with large numbers and small deltas.
  *
  * no way around this,
  * the int range and the float ranges are fundamentally different.
  *
  *
  * The integer range, is replaced by views, thats always what you want to do anyways.
  * For the float range speed matters less than consistency,
  *
*/



/**
  * what is a range ?
  * a sequence of numbers specified by [start stop[ and delta?
  * what does reversing it mean? [stop start[ delta, or the opposed order for the values of the original range?
  *
  * Why use ranges? unclear,
  * reverse iterators are difficult
  *
  * as input to e.g. a function for sampling something without needing
  *  to explicitly create the vector of values first
  *
  * range as a view of a generating function? this holds promise!
  */
template<class T>
struct gen_view{
    const T gen;
    int current;
    gen_view(T gen, int start, int stop):gen(gen), current(0){}
    gen_view begin() const { return gen_view(gen, start, stop); }

    struct range_end    {        T stop;    };
    range_end end()     { return range_end{stop}; }
    auto operator*() const { return gen(i);}
    range& operator++()    {current++;return *this;  }
    // compared to this in for each
    bool operator != (range_end end){return current != stop;}

};
struct genex{
    double start;
    double delta;
    operator(int i){ return start + delta*i;}
};


struct range{
    range(f);
    int start, delta, stop;
    rangei(int start, int  delta, int  stop):start(start),delta(delta),stop(stop){}

    const range& begin(){  return *this;}
    struct range_end    {        T stop;    };
    range_end end()     { return range_end{stop}; }
    T operator*() const {
        constexpr if () return start+(index*delta);
    }
    bool operator != (range_end end){return current<range_end.stop;}
};


#if 1


class range
{
public:
    int start, delta, stop;
    rangei(int start, int  delta, int  stop):start(start),delta(delta),stop(stop){}

    const range& begin(){  return *this;}
    struct range_end    {        T stop;    };
    range_end end()     { return range_end{stop}; }
    T operator*() const {
        constexpr if () return start+(index*delta);
    }
    bool operator != (range_end end){return current<range_end.stop;}
}



class range
{
    double start, delta;
    uint index;
    uint eindex;
    // must not include stop!
    // reverse will actually be different numbers, crap...
    range(double start, double delta=1, double stop):start(start), delta(delta), index(0) {}
    uint endindex(){
        if(start<stop && delta >0)
            return (stop - start)/ delta; // rounded down,
        if (start>stop && delta < 0)
            endindex=(start - stop)/ delta;
    }



    // forward iterator only? could work for ints, but floats become inconsistent.
    const range& begin(){  return *this;}
    struct range_end {        T stop;    };
    range_end end()  { return range_end{stop}; }
    T operator*() const{
        constexpr if () return start+(index*delta);
    }
    bool operator != (range_end end){return current<range_end.stop;}
    // integer only ones! or float with multiplication variant,
    // hmm in very fast code I will probably prefer to code with lower abstractions anyways, so lets use it.
    uint size()   { }
};

range sample(T start, T stop, uint samples){}
irange

#endif

constexpr void testsiome(){
    assert(false); // there is a claim that assertions in constexpr context are tested compiletime!
}

class test{
    void good(){
        testsiome();
        std::vector<double> ds;

        for(int i:ds); //
        /*
        range(0,1,10); // equivalent to
        range(0,10);
        range(10,-1,0);// from 9 to 0?
        for(int i: range(0,10)){
            cout<<i<<endl; // should give 0,1,...,9
        }
        assert(range(0,10).size()==10); // 10
        for(uint i:range(0, 10)){ // ok
            cout<<i<<endl;
        }
        uint i=10;
        range(-1, i); // should not work?
*/
        for (int i: generators::positive_numbers_mod_uintmax()){
            cout<<i<<endl;
        }
    }
} test2e576453;


} // end namespace cvl
#endif
