#pragma once
/* ********************************* FILE ************************************/
/** \file    epipolar_geometry.h
 *
 * \brief    This header contains fast and convenient versions of various common epipolar geometry functions
 * \remark
 * - c++11
 * - no dependencies
 * - header only
 *
 *
 *
 * All functions in this file use the following definitions:
 * - Camera Matrix C=K*[R|t]
 * - E= (K1^T)*F*K
 * - F=K.inv()*E*(K1^T).inv()
 * - K=K1 if ommited
 * - E=(t_{cross})R exactly ie not scale normalized!
 * Note:
 * - E has 6 dof, but can be considered a homogeneous element => 5 dof
 *
 *
 *
 *
 *
 *
 * \todo
 * - this code is replicated more frequently than any other but this is where it should be...
 *
 *
 *
 *
 *
 * \author   Mikael Persson
 * \date     2014-04-01
 * \note MIT licence
 *
 ******************************************************************************/
#include <mlib/utils/cvl/matrix.h>




namespace cvl{

template<class T>
/**
 * @brief getResampledK
 * @param K0 standard form 4 value linear intrinsic matrix assumes col,row,1
 * @param col_scale smaller than 1 for subsampling
 * @param row_scale
 * @return
 */
Matrix<T,3,3> getResampledK(const Matrix<T,3,3>& K0 ,
                                              const double& row_scale,
                                              const double& col_scale){

    //assert();// the matrix must be a standard form K
    return Matrix<T,3,3>(col_scale*K0(0,0),0,col_scale*K0(0,2),
                         0,row_scale*K0(1,1),row_scale*K0(1,2),
                         0,0,1);
}







/**
 * @brief epilineDistance
 * @param F The essential matrix
 * @param p the pinhole normalized camera measurement in camera P which has [R|t]
 * @param q the pinhole normalized camera measurement in camera Q
 *
 * This method should work for F and H aswell, just use pinhole measurement instead!
 * \todo
 * - unclear if that is true, write test!
 *
 * pFq=0 for p=proj(K[R|t]x), q=proj(K[I|0]x)
 *
 *
 *
 */
template<class T>
T epilineDistance(const  Matrix3<T>& F, const  Vector2<T>& p,const  Vector2<T>& q){
    /*
     * return fabs(p.dot((F*q.homogeneous()).lineNormalize()));
     *
     */
    // equivalent to, but faster 15%
    T l1=F(0,0)*q[0] + F(0,1)*q[1] + F(0,2);
    T l2=F(1,0)*q[0] + F(1,1)*q[1] + F(1,2);
    T ls=T(1.0)/std::sqrt(l1*l1 + l2*l2);// cos(a)^2 +sin(a)^2 ska vara 1 ie skalning,
    T l3=F(2,0)*q[0] + F(2,1)*q[1] + F(2,2);
    T v=(l1*p[0] + l2*p[1] + l3)*ls;
    if(v<0)v=-v;
    return  v;

    /*
          Vector3d p =  Vector3d( pun[0], pun[1], 1.0);     Vector3d q =  Vector3d( qun[0], qun[1], 1.0);

          Vector3d l = E * q;    l = 1.0 / sqrt(l[0]*l[0] + l[1]*l[1]) * l;
         return fabs(  dot(p, l) );
    */
}
template<class T> Vector3<T> getEpiline(const Matrix3<T>& F, const Vector2<T>& q){
    //  F=KEKinv
    return (F*(q.homogeneous())).lineNormalize();

}
} // end namespace cvl
