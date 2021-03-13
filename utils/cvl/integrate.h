#pragma once
/* ********************************* FILE ************************************/
/** \file    integrate.h
 *
 * \brief    This header contains a general numeric integrator, with optional upper and lower bounds
 *
 * \remark
 * - c++11
 * - no dependencies
 * - header only
 *

 * \author   Mikael Persson
 * \date     2020-04-01
 * \note MIT licence
 *
 ******************************************************************************/
#include <vector>
#include <algorithm>
namespace cvl
{

template<class function> long double
integrate_fast(const function& f,
               long double from,
               long double to,
               unsigned int steps=1e9){
    if(to<=from) return 0;
    long double span=to-from;
    long double delta=span/(long double)(steps);
    long double val=0;
    for(unsigned int i=0;i<steps;++i){
        long double time=from+(long double)(i)*delta;
        val+=f(time)*delta;
    }
    return val;
}
template<class function>
long double integrate_accurate(const function& f, long double from, long double to, unsigned int steps=1e9){
    if(to<=from) return 0;
    long double span=to-from;
    long double delta=span/(long double)(steps);
    std::vector<long double> stepvs;stepvs.reserve(steps);
    for(unsigned int i=0;i<steps;++i){
        long double time=from+(long double)(i)*delta;
        stepvs.push_back(f(time)*delta);
    }
    std::sort(stepvs.begin(),stepvs.end());
    long double val=0;
    for(auto v:stepvs) val+=v;
    return val;
}

}// en<T> namespace cvl



