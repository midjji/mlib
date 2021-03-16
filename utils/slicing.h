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
std::vector<std::vector<T>> slice_by_time(const std::vector<T>& ts, uint slices)
{

    if(ts.size()<2) return {ts};
    if(slices <2) return {ts};
    std::set<double> utimes;    for(const auto& t:ts)        utimes.insert(t.time);


    std::map<int, std::vector<T>> map;
    if(slices>=utimes.size()){
        for(auto& t:ts){
            map[t.time].push_back(t);
        }
    }
    else{
        // lets assume uniformely distributed...

        double span=*utimes.rbegin() - *utimes.begin();
        double delta=span/slices;
        // lets assume roughly evenly spread measurements
        for(auto& t:ts){
            auto& v=map[std::round((t.time - *utimes.begin())/delta)];
            v.reserve(ts.size());
            v.push_back(t);
        }
    }
    // sort them internally
    for(auto& [key,val]: map){
            std::sort(val.begin(),val.end(),[](const T& a, const T& b){return a.time<b.time;});
    }

    std::vector<std::vector<T>> ret;
    for(auto& [key,val]: map)
        ret.push_back(val);

    return ret;
}


}
