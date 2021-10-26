#pragma once
#include <map>
namespace cvl {
template <typename A, typename B>

/**
 * @brief A bijective map, i.e. each pair is unique.
 *
 */
struct BiMap{
    std::map<A,B> ab;
    std::map<B,A> ba;
    void insert(A a,B b){
        ab[a]=b;
    }

};
}
