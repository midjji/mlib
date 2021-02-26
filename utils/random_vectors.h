#pragma once
/* ********************************* FILE ************************************/
/** \file    random_vectors.h
 *
 * \brief   Uniform sampling on spheres is not trivially obvious, but very useful.
 *
 * \remark
 * - c++11
 * - no dependencies
 * - header only
 *
 *
 *
 *
 *
 * \author   Mikael Persson
 * \date     2010-04-01
 * \note MIT licence
 *
 ******************************************************************************/

#include <mlib/utils/random.h>
#include <mlib/utils/cvl/matrix.h>
#include <mlib/utils/cvl/pose.h>
namespace mlib{
/// Uniform distribution on the unit sphere
template<class T,int R> cvl::Vector<T,R> getRandomUnitVector(){

    cvl::Vector<T,R> n;

    for(int i =0;i<R;++i)
        n[i]=randn<T>(0,1);

    // can happen...
    if(n.abs().sum() <1e-10)
        return getRandomUnitVector<T,R>();
    return n.normalized();

}
template<class T> cvl::Pose<T> getRandomPose(){

return cvl::Pose<T>(getRandomUnitVector<T,4>(),getRandomUnitVector<T,3>()*randu<T>(0,10));

}

}
