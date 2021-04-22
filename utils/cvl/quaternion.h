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
#include <iostream>
#include <ceres/jet.h> // just no way around it...
#include <mlib/utils/cvl/matrix.h>


namespace cvl{

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
 * Tested by posetest.cpp
 */
template<class T>
class Quaternion
{
    // unit quaternion assumed!

public:
    T& operator[](uint index){
        assert(index<4);
        return q[index];
    }
    Vector4<T> q;
    T& operator()(int i){ return q(i); }
    const T& operator()(int i) const{ return q(i); }
    Quaternion()=default;

    Quaternion(Vector4<T> q):q(q){

        static_assert(std::is_trivially_destructible<Quaternion<double>>(),"speed");
        static_assert(std::is_trivially_copyable<Quaternion<double>>(),"speed");
        // the following constraints are good, but not critical
        static_assert(std::is_trivially_assignable<Quaternion<double>,Quaternion<double>>(),"speed");
        static_assert(std::is_trivially_copy_constructible<Quaternion<double>>(),"speed");
        static_assert(std::is_trivially_constructible<Quaternion<double>>(),"speed");
        static_assert(std::is_trivially_default_constructible<Quaternion<double>>(),"speed");
        static_assert(std::is_trivial<Quaternion<double>>(),"speed");
    }
    Quaternion(T w, Vector3<T> v):q(w,v[0],v[1],v[2]){}
    Quaternion(Vector<T,7> qt):q(qt[0],qt[1],qt[2],qt[3]){}
    Quaternion(Vector3<T> x):q(Vector4<T>(T(0.0),x[0],x[1],x[2])){}
    T real() const{return q[0];}

    Vector3<T> x(){
        // normalize?
        return Vector3<T>(q[1],q[2],q[3]);
    }
    Vector3<T> unit_rotate(Vector3<T> x)
    {
        Quaternion qxqc=(*this)*Quaternion(x)*conj();
        // simplify!
        ///TODO!
        //#warning "Simplify this eqn for a significant accuracy and speed boost!"

        //assert((qxqc[0]-T(0.0))<1e-10);
        return qxqc.x();
    }
    Quaternion conj() const
    {
        return Vector4<T>(q[0],-q[1],-q[2],-q[3]);
    }

    Vector3<T> imag_of_multiply(Quaternion b){
        // this is for when you only want the imaginary part,
        //like when you know the scalar part will be zero!
        // such as for the second mult of unit rotate, or omega or alpha
        Matrix<T,3,4> M(q(1),            q(0),        -q(3),         q(2),
                        q(2),           q(3),         q(0),        -q(1),
                        q(3),           -q(2),        q(1),         q(0));
        return M*b.q;
    }
    Vector3<T> imag_of_multiply_b_conj(Quaternion b){ // multiply by conjugate of b
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
     * atan2 must be ceres::atan2 if its to be derivable
     */
    T theta_(T cos_theta, T abs_sin_theta) const
    {
        if(abs_sin_theta<T(1e-5)) return cos_theta;
        T th=((cos_theta < T(0.0)) ?
                  ceres::atan2(-abs_sin_theta, -cos_theta):
                  ceres::atan2(abs_sin_theta, cos_theta));
        return th;
    }
    /**
     * @brief theta_
     * @return
     *
     * Slightly slower, but more consistent to just use one?
     */
    T theta_() const
    {
        T sin_squared_theta = vec().squaredNorm();
        T abs_sin_theta = ceres::sqrt(sin_squared_theta);
        //cout<<"sin_theta: "<<sin_theta<<endl;
        T cos_theta = real(); // note we dont use the /2 in the exponent etc.

        T th=((cos_theta < T(0.0)) ?
                  ceres::atan2(-abs_sin_theta, -cos_theta):
                  ceres::atan2(abs_sin_theta, cos_theta));
        return th;
    }




    Quaternion ulog() const {


        //if(ceres::abs(q.squaredNorm()-T(1))>T(1e-10)) std::cout<<"ulog: "<<q.squaredNorm()<<std::endl;

        //Vector4<T> q=this->q;        if(q[0]<T(0))            q=-q;

        const T& q0 = q[0];
        const T& q1 = q[1];
        const T& q2 = q[2];
        const T& q3 = q[3];

        const T sin_squared_theta = q1 * q1 + q2 * q2 + q3 * q3;
        if(sin_squared_theta<T(1e-10))
            return Vector4<T>(T(0),q1,q2,q3);




        const T sin_theta = ceres::sqrt(sin_squared_theta);
        //cout<<"sin_theta: "<<sin_theta<<endl;
        const T& cos_theta = q0;
        T theta =theta_(cos_theta, sin_theta);


        Vector<T,4> out;
        out[0]=T(0.0);
        T k=theta/sin_theta;
        //cout<<"theta: "<<theta<<endl;
        out[1] = q1 * k;
        out[2] = q2 * k;
        out[3] = q3 * k;


        return out;
    }

    Quaternion uexp(T alpha) const noexcept{
        if(q[0]*q[0]>T(1e-10))
            std::cout<<"u exp: "<<q[0]<<std::endl;
        auto v=vec();
        auto vv=v.squaredNorm();
        auto vn=ceres::sqrt(vv);


        return Quaternion(ceres::cos(vn*alpha),
                          ceres::sin(vn*alpha)*v/vv
                          );
    }


    // probably works for all values of alpha!
    //assumes unit quaternion,
    //since non unit quaternions technically have ambigious power
    Quaternion upow(double alpha) const{
        //if(ceres::abs(q.squaredNorm()-T(1))>T(1e-10))std::cout<<"upow: "<<q.squaredNorm()<<std::endl;

        //return ulog().uexp(T(alpha));

        //Vector4<T> q=this->q;        if(q[0]<T(0))            q=-q;
        //return expuq(log(q)*T(alpha));


        if(alpha==0.0) return Vector4<T>(T(1.0),T(0),T(0),T(0));
        if(alpha==1.0) {
            return q;
        }

        // special case

        const T& cos_theta = q[0];
        //const T& q0 = q[0];
        const T& q1 = q[1];
        const T& q2 = q[2];
        const T& q3 = q[3];
        const T sin_squared_theta = q1 * q1 + q2 * q2 + q3 * q3;

        // the square may be 0 even if q[0]!=1.0
        // due to rounding
        if(sin_squared_theta<T(1e-12)){
            //mlog()<<"hitting special case in pow: "<<sin_squared_theta<<"\n";
            //return *this;
            // lhospitals to get the limit..
            return Vector4<T>(T(1), q1*T(alpha),q2*T(alpha),q3*T(alpha));
        }

        Vector<T,4> out;
        // For quaternions representing non-zero rotation, the conversion
        // is numerically stable.

        const T sin_theta = ceres::sqrt(sin_squared_theta);

        T theta =theta_(cos_theta,sin_theta);



        out[0]= ceres::cos(alpha*theta);
        T k =   ceres::sin(alpha*theta) / sin_theta;

        out[1] = q1 * k;
        out[2] = q2 * k;
        out[3] = q3 * k;
        return out;
    }



    // using inclass as alternative may provide unfortunate casts
    Quaternion operator*(Quaternion<T> b){
        return QuaternionProductMatrix(q)*b.q;
    }
    // using inclass as alternative may provide unfortunate casts
    Quaternion operator*(T b){
        return q*b;
    }
    // using inclass as alternative may provide unfortunate casts
    Quaternion operator+(Quaternion b){
        return q+b.q;
    }
    // using inclass as alternative may provide unfortunate casts
    Quaternion& operator+=(Quaternion b){
        q+=b.q;
        return *this;
    }
    // using inclass as alternative may provide unfortunate casts
    Quaternion operator-(Quaternion b){
        return q-b.q;
    }
    Quaternion operator-(){
        return Quaternion(-q);
    }
    void normalize(){
        q/=ceres::sqrt(q.squaredNorm());
    }

    double norm(){
        return q.norm();
    }
    T squaredNorm(){
        return q.squaredNorm();
    }

    double geodesic_angle_degrees(Quaternion<double> b){
        return (180.0/3.1415)*(conj()*b).theta_()/2.0; // the last /2.0 is to convert it to half sphere?
    }

    // for rotations, the minimum distance!
    double geodesic(Quaternion<double> b){
        // dont use this for ceres, use geodesic_vector instead!
        return (conj()*b).ulog().x().norm();
    }
    Vector3<T> geodesic_vector(Quaternion<T> b){
        return (conj()*b).ulog().x();
    }
    T sign_minimizing_distance(Quaternion<T> b){
        T d0=(q - b.q).norm();
        T d1=(q + b.q).norm();
        if(d0<d1) return d0;
        return d1;
    }


    Vector3<T> vec() const{return Vector3<T>(q[1],q[2],q[3]);}
    T a() const{return q[0];}

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
}

