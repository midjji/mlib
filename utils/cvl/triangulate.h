#pragma once
/* ********************************* FILE ************************************/
/** \file   triangulate.h
 *
 * \brief    This header contains a linear fast symmetric triangulation method
 *
 * \remark
 * - c++11
 * - no dependencies
 * - header only
 * - cuda enabled
 * - assumes noise free data!
 *
 *
 * \todo
 * - add bias, variance compensated variants
 *
 *
 *
 *
 *
 * \author   Mikael Persson
 * \date     2015-04-01
 * \note MIT licence
 *
 ******************************************************************************/
#include <mlib/utils/cvl/matrix.h>
#include <mlib/utils/cvl/pose.h>
namespace cvl{
/**
 * @brief triangulate fast stereo triangulation
 * @param yln pinhole normalized measurement in the left camera assumed to be identity
 * @param yrn pinhole normalized measurement in the right camera assumed to be offset by (-baseline,0,0)
 * @param baseline
 * @param x
 *
 * ie x_right=Prl*x_left, Prl=Pose(Vector3(-baseline,0,0))
 *
 * If disparity <0, set disparity to nearly 0.
 *
 * No alloc, no throw, intended for double
 * restrict gives some minor speedup when available
 *
 * does not deal with behind the camera triangulation in a good way
 *
 * is not as good a approximation as the midpoint method
 *
 * should be followed by nl refinement
 *
 * __restrict__ does not work on clang
 */
template<class T> mlib_host_device_
void triangulate(const  T* yln, const  T* yrn, T baseline,  T* x){

    x[2]=yln[0] - yrn[0];
    if(x[2]<1e-7) x[2]=1e-7; // works for float and double, value should never be smaller than 1e-4 or its effectively 0 anyways
    x[2]=baseline/x[2];
    x[1]=(yln[1] + yrn[1])*0.5*x[2];
    x[0]=((yln[0] + yrn[0])*x[2] + baseline)*0.5;
}



/**
 * @brief triangulate fast stereo triangulation
 * @param yln pinhole normalized measurement in the left camera assumed to be identity
 * @param yrn pinhole normalized measurement in the right camera assumed to be offset by (-baseline,0,0)
 * @param baseline
 * @return the triangulated point
 * ie x_right=Prl*x_left, Prl=Pose(Vector3(-baseline,0,0))
 *
 * If disparity <0, set disparity to nearly 0.
 *
 */
template<class T> mlib_host_device_
Vector3<T> triangulate(const Vector2<T>& yln,const  Vector2<T>& yrn, T baseline){

    Vector3<T> x;
    triangulate<T>((&yln[0]),(&yrn[0]),baseline,(&x[0]));
    return x;
    /*
 * equivalent to, but faster than
    Vector3<T> x;
double disp=yln[0] - yrn[0];
    if(disp<1e-10) disp=1e-10;
    // symmetric cost
    x[2]=(baseline/disp);
    x[1]=(yln[1] + yrn[1])*0.5*x[2];
    x[0]=((yln[0] + yrn[0])*x[2] + baseline)*0.5;
    return x;
    */
}
/**
 * @brief triangulate
 * @param f
 * @param baseline
 * @param disparity
 * @return the triangulated depth
 */
template<class T>
mlib_host_device_
T triangulate(T f, T baseline, T disparity){
    return     f*baseline/disparity;
}
/**
 * @brief triangulate
 * @param yn
 * @param f
 * @param baseline
 * @param disparity
 * @return the triangulated pose
 */
template<class T>
mlib_host_device_
Vector3<T> triangulate(Vector2<T> yn, T f, T baseline, T disparity){
    Vector3<T> x;
    x[2]=triangulate(f,baseline,disparity);
    x[0]=yn[0]*x[2];
    x[1]=yn[1]*x[2];
    return x;
}
/**
 * @brief triangulate_ray
 * @param yn
 * @param f
 * @param baseline
 * @param disparity
 * @return the triangulated pose
 */
template<class T>
mlib_host_device_
inline Vector4<T> triangulate_ray(Vector2<T> yn,
                                          T fx, T baseline,
                                          T disparity){
    return {    yn[0],
                yn[1],
                T(1.0),
                disparity/(fx*baseline)};
}
template<class T>
mlib_host_device_
inline Vector4<T> triangulate_ray(Vector3<T> yn,
                                          T fx, T baseline){
    return {    yn[0],
                yn[1],
                T(1.0),
                yn[2]/(fx*baseline)};
}


template<class T>
bool LineLineIntersect( Vector3<T>& p1, Vector3<T>& p2,
                        Vector3<T>& p3,Vector3<T>& p4,
                        Vector3<T>& pa,Vector3<T>& pb,
                        T& mua, T& mub){
    Vector3<T> p13,p43,p21;
    T d1343,d4321,d1321,d4343,d2121;
    T numer,denom;

    p13[0] = p1[0] - p3[0];
    p13[1] = p1[1] - p3[1];
    p13[2] = p1[2] - p3[2];
    p43[0] = p4[0] - p3[0];
    p43[1] = p4[1] - p3[1];
    p43[2] = p4[2] - p3[2];
    if (fabs(p43[0]) < T(1e-8) && fabs(p43[1]) < T(1e-8) && fabs(p43[2]) < T(1e-8))
        return false;
    p21[0] = p2[0] - p1[0];
    p21[1] = p2[1] - p1[1];
    p21[2] = p2[2] - p1[2];
    if (fabs(p21[0]) < 1e-8 && fabs(p21[1]) < 1e-8 && fabs(p21[2]) < 1e-8)
        return false;

    d1343 = p13[0] * p43[0] + p13[1] * p43[1] + p13[2] * p43[2];
    d4321 = p43[0] * p21[0] + p43[1] * p21[1] + p43[2] * p21[2];
    d1321 = p13[0] * p21[0] + p13[1] * p21[1] + p13[2] * p21[2];
    d4343 = p43[0] * p43[0] + p43[1] * p43[1] + p43[2] * p43[2];
    d2121 = p21[0] * p21[0] + p21[1] * p21[1] + p21[2] * p21[2];

    denom = d2121 * d4343 - d4321 * d4321;
    if (fabs(denom) < 1e-8)
        return false;
    numer = d1343 * d4321 - d1321 * d4343;

    mua = numer / denom;
    mub = (d1343 + d4321 * (mua)) / d4343;

    pa[0] = p1[0] + mua * p21[0];
    pa[1] = p1[1] + mua * p21[1];
    pa[2] = p1[2] + mua * p21[2];
    pb[0] = p3[0] + mub * p43[0];
    pb[1] = p3[1] + mub * p43[1];
    pb[2] = p3[2] + mub * p43[2];

    return true;
}

template<class T>
/**
 * @brief triangulate
 * @param Ppw
 * @param Pqw
 * @param p pinholenormalized yn
 * @param q pinholenormalized yn
 *
 * return Xw such that x_p= Ppw*Xw, x_q=Pqw*Xw
 *
 * The method may fail.
 *
 *
 */
Vector3<T> triangulate(
        const Pose<T>& Ppw,
        const Pose<T>& Pqw,
        const Vector2<T>& p,
        const Vector2<T>& q){

    Matrix3<T> C0=Ppw.getR();
    Matrix3<T> C1=Pqw.getR();
    Vector3<T> tc0=Ppw.getT();
    Vector3<T> tc1=Pqw.getT();
    Vector3<T> c0 = Ppw.getTinW();
    Vector3<T> c1 = Pqw.getTinW();

    Vector2d x1=p;
    Vector2d x2=q;

    Vector3<T> pa, pb;
    double mua, mub;

    Vector3<T> x1c1;
    //
    //C0.transpose()*(x1.homogeneous() -tc0)

    x1c1[0] = C0(0,0) * (x1[0]-tc0[0]) + C0(1,0) * (x1[1]-tc0[1]) + C0(2,0) * (1.0-tc0[2]);
    x1c1[1] = C0(0,1) * (x1[0]-tc0[0]) + C0(1,1) * (x1[1]-tc0[1]) + C0(2,1) * (1.0-tc0[2]);
    x1c1[2] = C0(0,2) * (x1[0]-tc0[0]) + C0(1,2) * (x1[1]-tc0[1]) + C0(2,2) * (1.0-tc0[2]);

    Vector3<T> x2c2;
    x2c2[0] = C1(0,0) * (x2[0]-tc1[0]) + C1(1,0) * (x2[1]-tc1[1]) + C1(2,0) * (1.0-tc1[2]);
    x2c2[1] = C1(0,1) * (x2[0]-tc1[0]) + C1(1,1) * (x2[1]-tc1[1]) + C1(2,1) * (1.0-tc1[2]);
    x2c2[2] = C1(0,2) * (x2[0]-tc1[0]) + C1(1,2) * (x2[1]-tc1[1]) + C1(2,2) * (1.0-tc1[2]);

    LineLineIntersect(c0, x1c1, c1, x2c2, pa, pb, mua, mub);
    Vector3<T> X=pa + 0.5*(pb-pa);

    return X;
}










}// end namespace cvl
