#pragma once
/* ********************************* FILE ************************************/
/** \file    random.h
 *
 * \brief    This header contains contains convenience functions for repetable random values
 *
 * \remark
 * - c++11
 * - no dependencies
 * - self contained(just .h,.cpp)
 * - C++11 accurate generators are consistently faster and better than rand ever was!
 * - Repeatability is guaranteed if the seed is set and only these functions are used.
 *
 *
 *
 * \todo
 *
 * - convert to pure header for ease of inclusion and to ensure current not original flags are used!
 *
 *  Cmake Options: sets the flags
 * -DRANDOM_SEED_VALUE=0
 * -DRANDOM_SEED_FROM_TIME ON
 *  * option:
 *
 *
 *  * RANDOM DEFAULTS TO
 * RANDOM_SEED_VALUE 0
 * RANDOM_SEED_FROM_TIME OFF
 *
 *  Note, there is no way to have synch values for a multithreaded system.
 *
 *
 * \author   Mikael Persson
 * \date     2007-04-01
 * \note MIT licence
 *
 ******************************************************************************/
//////////////////// SELF CONTAINED ////////////////////////////




#include <random>
#include <vector>
#include <set>
#include <mutex>
#include <algorithm>
#include <assert.h>

namespace mlib{

namespace random{
// this is not thread safe, and it cant meaningfully be so either.
// no not even if you make one per thread
// no threaded program is repeatable, ever, regardless
// even in the single threaded case,
// you cannot use the generator in a static init, due to init order fiasco...

static const std::uint64_t seed{
#ifdef RANDOM_SEED_FROM_TIME
    std::chrono::system_clock::now().time_since_epoch().count()
#else
#ifndef RANDOM_SEED_VALUE
#define RANDOM_SEED_VALUE 0
#endif
RANDOM_SEED_VALUE
#endif
};
static std::default_random_engine generator(seed);
} // end namespace random

double randu(double low=0, double high=1);
int randui(int low=0, int high=1);
double randn(double mean=0, double sigma=1);

/**
 * \namespace mlib::random
 * \brief Contains convenient and repeatable random number functions see random.h
 *
 */
namespace random{


template<unsigned int size>
/**
 * @brief get_unit_vector
 * @return a unit vector which is uniformly distributed on the sphere
 */
std::array<double,size> random_unit_vector(){
    std::array<double,size> arr;
    for(auto& a:arr) a=0;
    auto len=[&]()->double{    double l=0;    for(auto& a:arr) l+=a*a; return l;       };
    double l;
    while(l<1e-6){ // possible num issues below this, practically never happens either
        for(auto& a:arr) a=randn(0,1);
        l=len();
    }
    l=std::sqrt(l);
    for(auto& a:arr)
        a/=l;
    return arr;
}

/**
 * @brief getNUnique returns the set with N ints between 0 and Max
 * @param N
 * @param Max highest value
 * @param ints
 */
template<class T> void getNUnique(uint N, T Max, std::set<T>& ints){
    assert(N < (uint)Max +1); // [0, Max]
    ints.clear();
    while (ints.size() < N) {
        ints.insert(randui(0, Max));
    }
}
/**
 * @brief getNUnique returns a vector with N ints between 0 and Max
 * @param N
 * @param Max
 * @param ints
 */
template<class T> void getNUnique(uint N, T Max,std::vector<T>& ints)
{
    std::set<T> set;
    getNUnique(N,Max,set);
    ints.clear();
    ints.reserve(set.size());
    for (auto i : set)        ints.push_back(i);
}

template<class T>
/**
 * @brief shuffle replacement for std::shuffle in order forcibly respect the choices
 * in the RANDOM_FLAGS in order to enable deterministic/repeatable behaviour
 * @param vec
 */
void shuffle(std::vector<T>& vec){
    std::vector<uint> indexes;indexes.reserve(vec.size());
    for(uint i=0;i<vec.size();++i){
        indexes.push_back(i);
    }
    std::shuffle(indexes.begin(),indexes.end(),generator);
    std::vector<T> tmp=vec;
    for(uint i=0;i<vec.size();++i)
        vec[i]=tmp[indexes[i]];
}

} // end namespace random

} // end namespace mlib
