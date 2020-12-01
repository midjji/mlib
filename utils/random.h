#pragma once
#ifndef random_h
#define random_h
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
#ifndef RANDOM_SEED_VALUE
#define RANDOM_SEED_VALUE 0
#endif




//////////////////// SELF CONTAINED ////////////////////////////




#include <random>
#include <vector>
#include <set>
#include <mutex>
#include <algorithm>
#include <assert.h>

namespace mlib{




namespace random{

/// the random generator mutex,
static std::mutex gen_mtx;
static std::default_random_engine generator;
static bool seeded=false;


template<int V>  void init_common_generator(){
    if(seeded) return; // dont lock unless you need to
    std::unique_lock<std::mutex> ul(gen_mtx); // lock
    if(seeded) return; // what if someone fixed it in the mean time?

    generator=std::default_random_engine();
    seeded=true;

#ifdef RANDOM_SEED_FROM_TIME
    unsigned long int seed=static_cast<unsigned long>(
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count());
    generator.seed(seed);
#else
    generator.seed(RANDOM_SEED_VALUE);
#endif

    /// I have avoided using local static since that does not work for old compilers or sometimes new ones...
}


} // end namespace random
/**
 * @brief randu integer random value drawn from uniform distribution
 * @param low  - included
 * @param high - included
 * @return random value according to compile time random value strategy
 */
template<class T> T randu(T low=0, T high=1){
    static_assert(std::is_floating_point<T>::value,          "template argument not a floating point type");
    random::init_common_generator<0>();
    std::uniform_real_distribution<T> rn(low,high);
    return rn(random::generator);
}
/**
 * @brief randui integer random value drawn from uniform distribution
 * @param low  - included
 * @param high - included
 * @return random value according to compile time random value strategy
 */
template<class T> T randui(T low=0, T high=1){
    random::init_common_generator<0>();
    std::uniform_int_distribution<T> rn(low,high);
    return rn(random::generator);
}
/**
 * @brief randn random value drawn from normal distribution
 * @param m
 * @param sigma
 * @return random value drawn from normal distribution
 */
template<class T> T randn(T mean=0, T sigma=1){
    static_assert(std::is_floating_point<T>::value,          "template argument not a floating point type");
    random::init_common_generator<0>();
    std::normal_distribution<T> rn(mean, sigma);
    return rn(random::generator);
}


template<class T, unsigned int size>
/**
 * @brief get_unit_vector
 * @return a unit vector which is uniformly distributed on the sphere
 */
std::array<T,size> random_unit_vector(){
    static_assert(std::is_floating_point<T>::value,          "template argument not a floating point type");
    // also should not be a complex type...
    std::array<T,size> arr;
    for(uint64_t i=0;i<size;++i)
        arr[i]=randn<T>(0,1);
    T len=0;
    for(uint64_t i=0;i<size;++i)
        len+=randn<T>(0,1)*randn<T>(0,1);
    len=std::sqrt(len);
    for(uint64_t i=0;i<size;++i)
        arr[i]/=len;
    return arr;
}


































/**
 * \namespace mlib::random
 * \brief Contains convenient and repeatable random number functions see random.h
 *
 */
namespace random{



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
        ints.insert(randui<T>(0, Max));
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

    random::init_common_generator<0>();
    std::shuffle(indexes.begin(),indexes.end(),generator);

    std::vector<T> tmp=vec;
    for(uint i=0;i<vec.size();++i)
        vec[i]=tmp[indexes[i]];
}







/**
 * @brief getMinRansacIterations
 * @param p_inlier                  probability a point is a inlier
 * @param p_failure                 how unlikely should a failure be
 * @param model_points              how many in min case
 * @param sample_points             how many sample points are there, low numbers require a small increase
 * @param p_good_given_inlier       how many inliers are too distorted by noise
 * @return
 */
template<class T> T getMinRansacIterations(T p_inlier,
                              T p_failure,
                              uint model_points,
                              uint sample_points,
                              T p_good_given_inlier)
{
    static_assert(std::is_floating_point<T>::value,
          "template argument not a floating point type");
    assert(p_inlier>0);
    assert(p_inlier<1);
    assert(p_failure>0);
    assert(p_failure<1);
    // this is the range in which the approximation is resonably valid.
    p_inlier  = std::min(std::max(p_inlier,1e-2),1-1e-8);
    p_failure = std::min(std::max(p_failure,1e-8),0.01);

    double p_good=p_good_given_inlier*std::pow(p_inlier,(double)model_points);
    // always draw atleast +50? yeah makes it better
    double N=std::ceil((log(p_failure)/log(1.0-p_good)));

    // approximate hyp as bin
    // approximate bin as norm
    //=> a min number ~50-100 needed for it to be ok.
    if(sample_points<100){
        N=N+50; // actually depends on the model_points,
    }
    // p_good_given_inlier should be drop with increasing model points too, or noise aswell

    if(N<50 ) N=50;
    return N+10;
}








} // end namespace random

} // end namespace mlib
#endif //random_h

