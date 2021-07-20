#pragma once
/* ********************************* FILE ************************************/
/** \file    quaternion.h
 *
 * \brief    This header contains the quaternion class, for quaternion pow and log
 *
 * \remark
 * - c++11
 * - no dependencies
 * - header only
 * - cuda enabled
 *
 *
 *
 * \todo
 *
 * \author   Mikael Persson
 * \date     2020-04-01
 * \note MIT licence
 *
 ******************************************************************************/

#include <cmath>
#include <iostream>

#include <mlib/utils/cvl/matrix.h>
#include <mlib/utils/mlog/log.h>
#include <mlib/utils/cvl/quaternion_functions.h>

namespace cvl {


/**
 * @brief The Quaternion<T> class
 * Quaternion
 *
 *
 *
 *
 *  In the following, the Quaternions are laid out as 4-vectors, thus:
   q[0]  scalar part.
   q[1]  coefficient of i.
   q[2]  coefficient of j.
   q[3]  coefficient of k.
   for unit q:
 q=(cos(alpha/2),sin(alpha/2)N)
 N= rotation axis
 */
template<class T>
class Quaternion
{

public:
    // unit quaternion assumed!
    Vector4<T> q;
    inline T& operator[](int index){
        assert(index<4);
        return q[index];
    }


    inline T& operator()(int i){ return q(i); }
    inline const T& operator()(int i) const{ return q(i); }
    Quaternion()=default;

    Quaternion(Vector4<T> q):q(q){}
    Quaternion(T w, Vector3<T> v):q(w,v[0],v[1],v[2]){}
    Quaternion(Vector<T,7> qt):q(qt[0],qt[1],qt[2],qt[3]){}
    Quaternion(Vector3<T> x):q(Vector4<T>(T(0.0),x[0],x[1],x[2])){}
    inline T real() const{return q[0];}

    inline Vector3<T> x(){
        // normalize?
        return Vector3<T>(q[1],q[2],q[3]);
    }
    inline Vector3<T> unit_rotate(Vector3<T>& x)
    {
        return ((*this)*Quaternion<T>(x)*conj()).x();
    }
    inline Quaternion conj() const
    {
        return Vector4<T>(q[0],-q[1],-q[2],-q[3]);
    }

    inline Vector3<T> imag_of_multiply(Quaternion b){
        // this is for when you only want the imaginary part,
        //like when you know the scalar part will be zero!
        // such as for the second mult of unit rotate, or omega or alpha
        Matrix<T,3,4> M(q(1),            q(0),        -q(3),         q(2),
                        q(2),           q(3),         q(0),        -q(1),
                        q(3),           -q(2),        q(1),         q(0));
        return M*b.q;
    }
    inline Vector3<T> imag_of_multiply_b_conj(Quaternion b){ // multiply by conjugate of b
        // this is for when you only want the imaginary part,
        //like when you know the scalar part will be zero!
        // such as for the second mult of unit rotate, or omega or alpha
        Matrix<T,3,4> M(q(1),            -q(0),        q(3),         -q(2),
                        q(2),           -q(3),         -q(0),        +q(1),
                        q(3),           +q(2),        -q(1),         -q(0));
        return M*b.q;
    }


    /**
     * @brief theta_
     * @param cos_theta
     * @param abs_sin_theta
     * @return
     *
     * theta is the angle with represents the shortest path from this to the next
     *
     *
     *
     * atan2 must be ceres::atan2 if its to be derivable
     *
     *
     *
     *
     *
     */
    inline T theta_(T cos_theta, T abs_sin_theta) const {        return unit_quaternion::theta(cos_theta, abs_sin_theta);    }
    /**
     * @brief theta_
     * @return
     *
     * Slightly slower,
     */
    inline T theta_() const
    {
       return unit_quaternion::theta(q);
    }

    void warn_on_not_unit() const noexcept{

        #ifndef NDEBUG

        #endif
    }

    inline Quaternion ulog() const {
        warn_on_not_unit();
        return unit_quaternion::log(q);
    }
    // probably works for all values of alpha!
    //assumes unit quaternion,
    //since non unit quaternions technically have ambigious power
    inline Quaternion upow(double alpha) const{
        warn_on_not_unit();
        return unit_quaternion::pow(q,alpha);
    }



    // using inclass as alternative may provide unfortunate casts
    inline Quaternion operator*(Quaternion<T> b){
        return QuaternionProductMatrix(q)*b.q;
    }
    // using inclass as alternative may provide unfortunate casts
    inline Quaternion operator*(T b){
        return q*b;
    }
    // using inclass as alternative may provide unfortunate casts
    inline Quaternion operator+(Quaternion b){
        return q+b.q;
    }
    // using inclass as alternative may provide unfortunate casts
    inline Quaternion& operator+=(Quaternion b){
        q+=b.q;
        return *this;
    }
    // using inclass as alternative may provide unfortunate casts
    inline Quaternion operator-(Quaternion b){
        return q-b.q;
    }
    inline Quaternion operator-(){
        return Quaternion(-q);
    }
    inline void normalize(){
        unit_quaternion::normalize_quaternion(q);
    }

    inline double norm(){
        return q.norm();
    }
    inline T squaredNorm(){
        return q.squaredNorm();
    }
    inline double geodesic_angle_degrees(Quaternion<double> b){
        return (180.0/3.1415)*(conj()*b).theta_()/2.0; // the last /2.0 is to convert it to half sphere?
    }
    // for rotations, the minimum distance!
    inline double geodesic(Quaternion<double> b){
        // dont use this for ceres, use geodesic_vector instead!
        return (conj()*b).ulog().x().norm();
    }
    inline Vector3<T> geodesic_vector(Quaternion<T> b){
        return (conj()*b).ulog().x();
    }
    inline Vector3<T> vec() const{return Vector3<T>(q[1],q[2],q[3]);}
    inline T a() const{return q[0];}
};

// required for
// rotations and their derivatives, including w?
// derivative keeps the sign of the orig...
// returns error to not square it again in resid

template<uint N, class T>
Vector<T,N> sign_compensated_error(Vector<T,N> a, Vector<T,N> b)
{


    Vector<T,N> amb=a-b;
    Vector<T,N> apb=a+b;
    // min |a-b|,|a+b|
    // make this fast by using only the scalar part later
    if(amb.squaredNorm()<apb.squaredNorm())
        return amb;
    return apb;
}
template<class T>
std::ostream& operator<<(std::ostream& os,
                         cvl::Quaternion<T> q){
    return os<<"("<<q.q[0]<<", "<<q.q[1]<< " "<<q.q[2]<<" "<<q.q[3]<<")";
}



static_assert(std::is_trivially_destructible<Quaternion<double>>(),"speed");
static_assert(std::is_trivially_copyable<Quaternion<double>>(),"speed");
// the following constraints are good, but not critical
static_assert(std::is_trivially_assignable<Quaternion<double>,Quaternion<double>>(),"speed");
static_assert(std::is_trivially_copy_constructible<Quaternion<double>>(),"speed");
static_assert(std::is_trivially_constructible<Quaternion<double>>(),"speed");
static_assert(std::is_trivially_default_constructible<Quaternion<double>>(),"speed");
static_assert(std::is_trivial<Quaternion<double>>(),"speed");
}// end namespace cvl

