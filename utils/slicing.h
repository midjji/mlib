#pragma once
/* ********************************* FILE ************************************/
/** \file    slicing.h
 *
 * \brief    This header constains simple slicing functionality,
 *
 * \remark
 * - c++11
 * - no dependencies
 * - header only
 *
 * \todo
 * - add slice_view iterators
 * - rewrite to be generic to container type
 *
 *
 *
 * \author   Mikael Persson
 * \date     2014-04-01
 * \note BSD licence applies to this file alone
 *
 ******************************************************************************/
#include <vector>
#include <set>
#include <algorithm>
#include <map>
#include <cmath>
#include <mlib/utils/mlog/log.h>
namespace cvl{
#if 0
template<class T>
struct SliceView{
    // does not own, and does not guarantee vals keeps existing...
    std::vector<T>& vals;
    std::vector<
    T& operator()(int i, int j){ // should be variadic...

    }

};
#endif

template<class T>
//ts([i0:i1))
std::vector<T> slice(const std::vector<T>& ts, uint i0, uint i1)
{
    if(i0>=i1) return {};
    if(i1>ts.size()) i1=ts.size();
    std::vector<T> out;out.reserve(i1-i0);
    for(uint i=i0;i<i1;++i)out.push_back(ts[i]);
    return out;
}


template<class T>

std::vector<std::vector<T>> slice(
        const std::vector<T>& ts,
        const std::set<uint>& indexes)
{

    if(indexes.size()<2)
        return {ts};
    // [indexes[0], index[1]),
    // [indexes[1], indexes[2]),  ...
    // slices with indexes above end are included as empty


    std::vector<uint> index;
    index.reserve(indexes.size());
    for(uint i:indexes){
        if(i<ts.size())
            index.push_back(i);
        else{
            index.push_back(ts.size());
            break;
        }
    }


    std::vector<std::vector<T>> ret;ret.reserve(index.size());
    for(uint i=1;i<index.size();++i){
        ret.push_back(slice(ts,index[i-1], index[i]));
    }


    return ret;
}
template<class T>
std::vector<std::vector<T>> slice(const std::vector<T>& ts, uint slices){
    // assume sorted, assume even slices
    if(slices <2) return {ts};
    double delta=double(ts.size())/double(slices);
    std::vector<std::vector<T>> out;
    for(uint i=0;i<ts.size();++i){
        uint index=i/delta;
        if(index>=out.size()){
            out.push_back({});
            out.reserve(ts.size());
        }
        out.back().push_back(ts[i]);
    }
    return out;
}

template<class T>
std::vector<std::vector<T>> slice_by_time(std::vector<T> ts, uint slices)
{
    //std::sort(ts.begin(),ts.end(),[](const T& a, const T& b){return a.time_seconds()<b.time_seconds();});

    if(ts.size()<2) return {ts};
    if(slices <2) return {ts};
    std::set<double> utimes;    for(const auto& t:ts)        utimes.insert(t.time_seconds());


    std::map<double, std::vector<T>> map;
    if(slices>=utimes.size()){
        mlog()<<"split by time\n";
        for(auto& t:ts){
            map[t.time_seconds()].reserve(ts.size());
            map[t.time_seconds()].push_back(t);
        }
    }
    else{
        mlog()<<"split by time\n";
        // lets assume uniformely distributed...
        double t0=*utimes.begin();
        double t1=*utimes.rbegin();        
        double span= t1- t0;
        double delta=span/slices;
        // lets assume roughly evenly spread measurements
        for(auto& t:ts){
            auto& v=map[std::round((t.time_seconds() - t0)/delta)];
            v.reserve(ts.size());
            v.push_back(t);
        }
    }
    std::vector<std::vector<T>> ret;
    for(auto& [key,val]: map)
        ret.push_back(val);

    return ret;
}


}
