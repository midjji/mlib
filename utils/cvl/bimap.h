#pragma once
#include <map>
namespace cvl {
template <typename A, typename B>
struct BiMap{
    std::map<A,B> ab;
    std::map<B,A> ba;
};
}
