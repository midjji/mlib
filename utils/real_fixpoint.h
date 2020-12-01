#pragma once
/* ********************************* FILE ************************************/
/** \file    real_fixpoint.h
 *
 * \brief    This header contains a size templated fixpoint real,
 *
 * \remark
 * - c++11
 * - for large number accuracy
 * - unlike the usual dynamic bignum libs, this should support cuda, thanks to templating...
 * - should not have any undefined behaviour,
 * - slow is fine.
 *
 * \todo
 * - actually implement the basic operations,
 * - implement a test using gmp bignum
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
namespace cvl{

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
