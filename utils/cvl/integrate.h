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
#include <array>
#include <thread>

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


template<class function> long double
integrate_fast_mp_impl(unsigned int i0,
                       unsigned int i1,
                       unsigned int steps,
                       long double delta,
                       long double from,
                       long double& val,
                       const function& f){
    for(unsigned int i=i0;i<i1 && i<steps;++i){
        long double time=from+(long double)(i)*delta;
        val+=f(time)*delta;
    }
    return val;
}
template<class function> long double
integrate_fast_mp(const function& f,
                  long double from,
                  long double to,
                  unsigned int steps=1e9){
    if(to<=from) return 0;
    long double span=to-from;
    long double delta=span/(long double)(steps);


    constexpr int threads=16;
    unsigned int num=(steps +15)/threads;
    std::array<long double, threads> vals;
    for(auto& val:vals) val=0;
    std::array<std::thread, threads> thrs;
    for(int i=0;i<threads;++i){
        thrs[i]=std::thread(integrate_fast_mp_impl<function>,i*num, (i+1)*num, steps, delta, from, std::ref(vals[i]),std::ref(f));
    }
    for(auto& thr:thrs)
        thr.join();
    long double val=0;
    for(auto& v:vals)
        val+=v;


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
    std::sort(stepvs.begin(),stepvs.end(),[](double a, double b){return std::abs(a)<std::abs(b);});
    long double val=0;
    for(auto v:stepvs) val+=v;
    return val;
}

}// en<T> namespace cvl



