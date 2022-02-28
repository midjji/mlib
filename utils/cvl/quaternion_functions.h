#pragma once
#include <mlib/utils/cvl/matrix.h>
//#include <ceres/jet.h> // we dont want this everywhere, and mlib should not require ceres...
// unfortunately, some std:: functions incorrectly match the ceres::jet
// so forward declare what we will need, these will be defined when this is used in such context.
namespace ceres {
template <typename T, int N> struct Jet;
template <typename T, int N> Jet<T, N> atan2(const Jet<T, N>& g, const Jet<T, N>& f);
template <typename T, int N> Jet<T, N> sqrt(const Jet<T, N>& f);
template <typename T, int N> Jet<T, N> acos(const Jet<T, N>& f);
template <typename T, int N> Jet<T, N> cos(const Jet<T, N>& f);
template <typename T, int N> Jet<T, N> sin(const Jet<T, N>& f);
}


namespace cvl{
namespace unit_quaternion{


// must be fully available for optimal performance.
inline double theta(double cos_theta, double abs_sin_theta)
{
        return ((cos_theta < (0.0)) ?
                    std::atan2(-abs_sin_theta, -cos_theta):
                    std::atan2(abs_sin_theta, cos_theta));
}

template <typename V, int N> inline
ceres::Jet<V, N> theta(const ceres::Jet<V, N>& cos_theta, const ceres::Jet<V, N>& abs_sin_theta)
{
    using T=ceres::Jet<V, N>;
    return ((cos_theta < T(0.0)) ?
                ceres::atan2(-abs_sin_theta, -cos_theta):
                ceres::atan2(abs_sin_theta, cos_theta));
}
// must be fully available for optimal performance.
inline double theta(const Vector4d& q) {

    return theta(q[0], std::sqrt(q[1]*q[1] + q[2]*q[2] +q[3]*q[3]));
}
template <typename V, int N> inline
ceres::Jet<V, N> theta(const Vector4<ceres::Jet<V, N>>& q)
{
    return theta(q[0],ceres::sqrt(q[1]*q[1] + q[2]*q[2] +q[3]*q[3]));
}

inline Vector4<double> log(const Vector4d& q)
{
    using T=double;


    const T& q0 = q[0];
    const T& q1 = q[1];
    const T& q2 = q[2];
    const T& q3 = q[3];

    const T sin_squared_theta = q1 * q1 + q2 * q2 + q3 * q3;
    const T abs_sin_theta = std::sqrt(sin_squared_theta);
    const T& cos_theta = q0;
    T theta_ =theta(cos_theta, abs_sin_theta);


    Vector<T,4> out;
    out[0]=T(0.0);
    T k;
    // so this should be the short path
    if(abs_sin_theta<T(1e-12) )
        k = cos_theta<T(0) ? T(-1): T(1);
    else
        k = theta_/abs_sin_theta;
    //cout<<"theta: "<<theta<<endl;
    out[1] = q1 * k;
    out[2] = q2 * k;
    out[3] = q3 * k;

    return out;
}
inline
Vector4d pow(const Vector4d& q, double alpha)
{
    using T=double;
    if(alpha==0.0) return Vector4<T>(T(1.0),T(0),T(0),T(0));
    if(alpha==1.0) {            return q;        }


    // special case

    const T& cos_theta = q[0];
    //const T& q0 = q[0];
    const T& q1 = q[1];
    const T& q2 = q[2];
    const T& q3 = q[3];
    const T sin_squared_theta = T(1.0) - cos_theta*cos_theta;//q1 * q1 + q2 * q2 + q3 * q3;

    if(sin_squared_theta<T(1e-12)) {

        double asign=1; if(alpha<0) asign=-1;
        if(cos_theta <T(0)) asign=-asign;
        return Vector4<T>(cos_theta,q1*T(asign),q2*T(asign),q3*T(asign));
    }

    Vector<T,4> out;
    // For quaternions representing non-zero rotation, the conversion
    // is numerically stable.

    T abs_sin_theta = std::sqrt(sin_squared_theta);
    // theta in [-pi/2 to pi/2] such that the shorter path is chosen
    T theta_ =theta(cos_theta,abs_sin_theta);

    out[0]= std::cos(alpha*theta_);
    T k;
    if(abs_sin_theta<T(1e-6) ){
        //mlog()<<"hits this"<<abs_sin_theta<<"\n";
        k = cos_theta<T(0) ? T(-alpha): T(alpha);
    }
    else
        k = std::sin(alpha*theta_) / abs_sin_theta;

    out[1] = q1 * k;
    out[2] = q2 * k;
    out[3] = q3 * k;
    return out;
}


template <typename V, int N>
inline Vector4<ceres::Jet<V, N>> log(const Vector4<ceres::Jet<V, N>>& q)
{
    using T=ceres::Jet<V, N>;
    const T& q0 = q[0];
    const T& q1 = q[1];
    const T& q2 = q[2];
    const T& q3 = q[3];

    const T sin_squared_theta = q1 * q1 + q2 * q2 + q3 * q3;
    const T abs_sin_theta = ceres::sqrt(sin_squared_theta);
    const T& cos_theta = q0;
    // theta is picked as the shorter path
    T theta_ =theta(cos_theta, abs_sin_theta);


    Vector<T,4> out;
    out[0]=T(0.0);
    T k;
    // so this should be the short path
    if(abs_sin_theta<T(1e-6) )
        k = cos_theta<T(0) ? T(-1): T(1);
    else
        k = theta_/abs_sin_theta;
    //cout<<"theta: "<<theta<<endl;
    out[1] = q1 * k;
    out[2] = q2 * k;
    out[3] = q3 * k;

    return out;
}

template <typename V, int N> inline
Vector4<ceres::Jet<V, N>> pow(const Vector4<ceres::Jet<V, N>>& q, double alpha)
{
    using T=ceres::Jet<V, N>;
    if(alpha==0.0) return Vector4<T>(T(1.0),T(0),T(0),T(0));
    if(alpha==1.0) {            return q;        }


    // special case

    const T& cos_theta = q[0];
    //const T& q0 = q[0];
    const T& q1 = q[1];
    const T& q2 = q[2];
    const T& q3 = q[3];
    const T sin_squared_theta = q1 * q1 + q2 * q2 + q3 * q3;

    if(sin_squared_theta<T(1e-12)) {

        double asign=1; if(alpha<0) asign=-1;
        if(cos_theta <T(0)) asign=-asign;
        return Vector4<T>(cos_theta,q1*T(asign),q2*T(asign),q3*T(asign));
    }

    Vector<T,4> out;
    // For quaternions representing non-zero rotation, the conversion
    // is numerically stable.

    T abs_sin_theta = ceres::sqrt(sin_squared_theta);
    // theta in [-pi/2 to pi/2] such that the shorter path is chosen
    T theta_ =theta(cos_theta,abs_sin_theta);

    out[0]= ceres::cos(alpha*theta_);
    T k;
    if(abs_sin_theta<T(1e-12) ){
        //mlog()<<"hits this"<<abs_sin_theta<<"\n";
        k = cos_theta<T(0) ? T(-alpha): T(alpha);
    }
    else
        k = ceres::sin(alpha*theta_) / abs_sin_theta;

    out[1] = q1 * k;
    out[2] = q2 * k;
    out[3] = q3 * k;
    return out;
}
inline void normalize_quaternion(Vector4d& q){
    q.normalize();
}
template <typename T, int N> inline
void normalize_quaternion(Vector4<ceres::Jet<T, N>>& q){
    q/=ceres::sqrt(q.squaredNorm());
}

}// end namespace unit_quaternion
}// end namespace cvl
