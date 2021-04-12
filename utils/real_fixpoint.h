#if 0
#pragma once
/* ********************************* FILE ************************************/
/** \file
 *
 * \brief    This header contains a size templated ints and fix_point reals
 *
 *  Note that out of the standards implemented types
 *
 *  sizeof(long double) ==16
 *
 *  This means that the long double is enough for all other basic types!
 *
 * \remark
 *
 *
 *
 *
 * \author   Mikael Persson
 * \date     2015-04-01
 * \note BSD licence
 *
 ******************************************************************************/
#include <string>
#include <array>
#include <limits>
#include <vector>
#include <cmath>
namespace cvl{


struct IntX{

    // c + b*2^32 + a*2^32^2
    std::vector<std::uint32_t> data;
    bool sign; // ==negative
    std::uint32_t p232=std::numeric_limits<std::uint32_t>::max();

    // the biggest


    IntX(std::int64_t a){
        sign=a<0;
        if(a<0) a=-a;
        std::uint64_t b=a;
        data.push_back(std::uint32_t(b));
        data.push_back(std::uint32_t(b/p232));
    }
    IntX(std::uint64_t a){
        data.push_back(std::uint32_t(a));
        data.push_back(std::uint32_t(a/p232));
    }
    using ld=long double; // sizeof(ld)==16

    IntX(long double a){
        sign=a<0;
        if(a<0) a=-a;
        while(a>=1){
            data.push_back(std::uint32_t(a));
            a/=p232;
        }
    }

    long double real() const{
        // add the smallest numbers first to make the minimum numerics error
        // this means not possible to just do one mult per block
            long double d=data[0];

            for(uint i=1;i<data.size();++i){
                // std::pow specializes if the second one is integer
                d+=ld(data[i])*std::pow(ld(p232),i);
            }
            return d;
    }
    // there is nothing bigger than int 64_t
    std::int64_t integer(){
        long int d=data[0];
        for(uint i=1;i<data.size();++i){
            d+=li(data[i])*std::pow(li(p232),i);
        }
        return d;
    }

    std::uint32_t& back(uint i){
        return data[Size-i-1];
    }
    constexpr uint size(){return Size;}

    template<uint N>
    // + upcasts to the bigger size, but can still overflow
    UInt<std::max(Size,N)> operator+(UInt<N> b) const{
        Uint<std::max(Size,N)> c=*this;
        return b+=c;
    }
};

// size is the number of bytes /4
template<uint Size>

struct UInt{
    // a*2^32^2 + b*2^32 +c
    std::array<std::uint32_t,Size> data;
    std::uint32_t p232=std::numeric_limits<std::uint32_t>::max();

    UInt(std::uint64_t a){
        for(auto& d:data) d=0;

        back(0) =a; // only keeps the least significant bits
        a=a/p232;
        back(1) = a;
    }
    UInt(double a){
        for(int i=0;i<Size;++i){
            back(i) = a;
            a/=p232;
        }
    }
    UInt(long double a){
        for(int i=0;i<Size;++i){
            back(i) = a;
            a/=p232;
        }
    }
    template<uint N>
    UInt(UInt<N> b){
        for(auto& d:data) d=0;
        for(int i=0;i<Size && i<N;++i)
            back(i)=b.back(i);
    }

    long double real() const{
            long double d=data[0];
            for(int i=1;i<Size;++i){
                d*=p232;
                d+=data[i];
            }
            return d;
    }
    long unsigned int integer(){
        long unsigned int d=data[0];
        for(int i=1;i<Size;++i){
            d*=p232;
            d+=data[i];
        }
        return d;
    }

    std::uint32_t& back(uint i){
        return data[Size-i-1];
    }
    constexpr uint size(){return Size;}

    template<uint N>
    // + upcasts to the bigger size, but can still overflow
    UInt<std::max(Size,N)> operator+(UInt<N> b) const{
        Uint<std::max(Size,N)> c=*this;
        return b+=c;
    }
};



template<unsigned int Major, unsigned int Minor> class RealFixedPoint{
    // double requires less than 34 so 17+17, note using 4,8,16 and so on on either is faster.

    uint32_t data[Major + Minor]; // or use bitvector?
    bool sign;
    bool overflow; // ans so on? perhaps a status int instead, later...

    RealFixedPoint operator+(RealFixedPoint& b){
        // every
        uint32_t maxval=-1;  // should be an adaptive to bits max uint...
        uint64_t overflow=0;
        RealFixedPoint out;

        for(int i=0;i<Major+Minor;++i){
            int index=Major+Minor-1-i;
            uint64_t tmp = uint64_t(data[index]) + uint64_t(b.data[index]) + overflow;
            uint64_t v=tmp;
            out.data[index]=v;
            overflow =(tmp - (uint64_t)v)>>8*sizeof(uint); // overflow!
        }
        // final overflow dissapears... // wrapping around gets wierd
        return out;
    }
};

}
#endif
