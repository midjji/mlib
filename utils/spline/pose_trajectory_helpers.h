#pragma once

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <mlib/utils/cvl/pose.h>
#include <mlib/utils/spline/coeffs.h>
#include <mlib/utils/cvl/quaternion.h>

namespace cvl {



template<class T, long unsigned int N> inline Vector<Vector<T,7>,N/*Degree+1*/>
to_state(const std::array<const T * const, N>& cpts)
{
    Vector<Vector<T,7>,N/*Degree+1*/> ps;
    for(uint i=0;i<N;++i)
        ps[i]=Vector<T,7>::copy_from(cpts[i]);
    return ps;
}


template<class T, uint N/*Degree+1*/> inline Vector<Vector3<T>,N>
translation_control_points(const Vector<Vector<T,7>,N/*Degree+1*/>& state)
{

    Vector<Vector3<T>,N> ps;
    for(uint i=0;i<N;++i)
        ps[i]=Vector3<T>(state[i][4],state[i][5],state[i][6]);
    return ps;
}
template<class T, uint N>
inline Vector<Quaternion<T>,N>
quaternion_control_points(const  Vector<Vector<T,7>,N/*Degree+1*/>& state)
{
    Vector<Quaternion<T>,N> ps;
    for(uint i=0;i<N;++i)
        ps[i]=state[i];
    return ps;
}



template<class T, uint N /*Degree+1*/> inline Vector3<T>
compute_translation(const Vector<Vector<T,7>,N/*Degree+1*/>& state,
                    const Vector<double,N>& ccbs)
{

    auto ts=translation_control_points(state);
    Vector<T,3> t=Vector<T,3>::Zero();
    if(ccbs[0]!=0.0)
        t=ts[0]*T(ccbs[0]);

    for(uint i=1;i<N;++i)
        if(ccbs[i]!=0)
            t+=(ts[i] - ts[i-1])*T(ccbs[i]);
    return t;
}
template<class T, uint N /*Degree+1*/> inline Vector3<T>
compute_translation(const Vector<Vector<T,7>,N/*Degree+1*/>& state,
                    const SplineBasisKoeffs& sbk,
                    int derivative=0)
{
    static_assert (N>=1," order is degree>0, +1" );
    return compute_translation(state, sbk.cbs<N-1>(derivative));

}


#if 1
template <class T, uint N> inline

/**
 * @brief compute_qdot
 * @param state
 * @param as
 * @param compute_qdt
 * @param compute_qdtdt
 * @return
 *
 * Computing higher derivatives of q requires computing the lower derivatives of q
 *
 */

Vector3<Quaternion<T>> compute_qdot(
        const Vector<Vector<T,7>, N/*Degree+1*/>& state,
        const Vector3<Vector<double, N>>& as,
        bool compute_qdt,
        bool compute_qdtdt)
{
    // extrapolation is taken care of before this function is reached by modifying
    // the state and the as.
    compute_qdt=compute_qdt||compute_qdtdt;
    // slightly faster, principially better, but not absolutely sure...

    /*
    if constexpr (N>2){
        Vector3<Vector<double,N-1>> tmp;
        for(int i=0;i<3;++i)
            tmp[i]=as[i].drop_first();

        if(as[0][1]==1.0)
            return compute_qdot<T,N-1>(state.drop_first(), tmp,compute_qdt, compute_qdtdt);
    }
    */



    Vector<Quaternion<T>, N> cpts = quaternion_control_points(state);
    Vector<Quaternion<T>, N> ds,das;
    Vector<Quaternion<T>, N> qs;


    ds[0]=cpts[0];// ds[0] is q0!
    for(uint i=1;i<N;++i) ds[i]=cpts[i-1].conj()*cpts[i];

    for(uint i=0;i<N;++i) das[i]=ds[i].upow(as[0][i]);
    qs[0]=das[0];
    for(uint i=1;i<N;++i) qs[i] = qs[i-1]*das[i];

    Vector3<Quaternion<T>> ret;
    ret[0]=qs[N-1];



    if(compute_qdt)
    {
        Vector<Quaternion<T>, N> ws, da_dts;
        Vector<Quaternion<T>, N> q_dts;
        for(uint i=0;i<N;++i) ws[i]=ds[i].ulog();

        for(uint i=0;i<N;++i)
            if(as[1][i] !=0.0)
                da_dts[i] = ws[i]*das[i]*T(as[1][i]);
            else
                da_dts[i].q=Vector<T,4>::Zero();

        q_dts[0] = da_dts[0];
        for(uint i=1;i<N;++i){
            q_dts[i] = q_dts[i-1]*das[i];
            if(as[1][i]!=0.0)
                q_dts[i] += qs[i]*ws[i]*T(as[1][i]);
        }
        ret[1]= q_dts[N-1];

        if(compute_qdtdt)
        {

            Vector<Quaternion<T>, N> q_dt_dts;
            q_dt_dts[0].q =Vector<T,4>::Zero();
            if(as[1][0] !=0.0){
                q_dt_dts[0]=q_dts[0]*ws[0]*T(as[1][0]);
                if(as[2][0]!=0.0)
                    q_dt_dts[0]+=ws[0]*das[0]*T(as[2][0]);
            }
            for(uint i=1;i<N;++i){
                q_dt_dts[i] = q_dt_dts[i-1]*das[i];
                if(as[1][i]!=0.0)
                {
                    q_dt_dts[i] += q_dts[i-1]*da_dts[i];
                    auto tmp=q_dts[i]*T(as[1][i]);
                    if(as[2][i]!=0.0)
                        tmp+=qs[i]*T(as[2][i]);
                    tmp=tmp*ws[i];
                    q_dt_dts[i] += tmp;
                }
            }
            ret[2]=q_dt_dts[N-1];
        }
    }
return ret;
}
#else

template <class T, uint N>
inline Vector3<Quaternion<T>> compute_qdot(
        Vector<Vector<T,7>,N/*Degree+1*/> state, // keep the explicit one to help compile error make more sense
        Vector3<Vector<double,N>> as, bool compute_qdt, bool compute_qdtdt)  {


constexpr int Degree=N-1;

    // extrapolation is taken care of before this function is reached by modifying
    // the state and the as.

    compute_qdt=compute_qdt||compute_qdtdt;


    Vector<Quaternion<T>, N> cpts = quaternion_control_points(state);
    Vector<Quaternion<T>, N> ds,das,da_dts,ws;
    Vector<Quaternion<T>, N> qs,q_dts,q_dt_dts;


    ds[0]=cpts[0];// ds[0] is q0!
    for(int i=1;i<N;++i) ds[i]=cpts[i-1].conj()*cpts[i];
    for(int i=0;i<N;++i) das[i]=ds[i].upow(as[0][i]);
    qs[0]=das[0];
    for(int i=1;i<N;++i) qs[i] = qs[i-1]*das[i];

    if(compute_qdt){

        for(int i=0;i<N;++i) ws[i]=ds[i].ulog();
        da_dts[0] = Vector4<T>(T(0),T(0),T(0),T(0));
        for(int i=1;i<N;++i) da_dts[i] = ws[i]*das[i]*T(as[1][i]);
        if(as[1][0]!=0)
            std::cout<<"as[1][0]"<<as[1][0]<<"\n";
        q_dts[0] = da_dts[0];
        for(int i=1;i<N;++i)
            q_dts[i] = q_dts[i-1]*das[i] + qs[i]*ws[i]*T(as[1][i]);
    }
    if(compute_qdtdt){
        // simplifiable



        if(as[0][0] !=1.0 || as[1][0] !=0.0||as[2][0] !=0.0){
            Quaternion<T> tmp(Vector4<T>(-ws[0].q.dot(ws[0].q)*T(as[1][0]*as[1][0]),
                    ws[0][0]*T(as[2][0]),
                    ws[0][0]*T(as[2][0]),
                    ws[0][0]*T(as[2][0])));// ws[1]*ws[1]
            q_dt_dts[0] = das[0]*tmp; // else its zero
        }
        // lots more simplification possible...
        for(int i=1;i<N;++i){

            q_dt_dts[i] =
                    q_dt_dts[i-1]*das[i] +
                    q_dts[i-1]*da_dts[i]+
                    q_dts[i]*ws[i]*T(as[1][i]) +
                    qs[i]*ws[i]*T(as[2][i]);
            /*
                    q_dt_dts[i] =
                            q_dt_dts[i-1]*das[i] +(
                            q_dts[i-1]*das[i]*T(as[1][i])+
                            q_dts[i]*T(as[1][i]) +
                            qs[i]*T(as[2][i]))*ws[i];
                            */
        }
    }
    return Vector3<Quaternion<T>>(qs[Degree],
                                  q_dts[Degree],
                                  q_dt_dts[Degree]);

}

#endif
template <class T, uint N> inline
Vector3<Quaternion<T>> compute_qdot(
        const Vector<Vector<T,7>, N/*Degree+1*/>& state,
        const SplineBasisKoeffs& sbk,
        bool compute_qdt,
        bool compute_qdtdt)
{
    static_assert (N>=1," order is degree>0, +1" );
    return compute_qdot(state, sbk.cbss<N-1,3>(), compute_qdt, compute_qdtdt);
}


template <class T, int N>
inline Quaternion<T> compute_quaternion(
        const Vector<Vector<T,7>,N/*Degree+1*/>& state,
        const Vector<double,N>& a)
{
    Vector<Quaternion<T>, N> cpts = quaternion_control_points(state);
    Vector<Quaternion<T>, N> ds,das;
    Vector<Quaternion<T>, N> qs;


    ds[0]=cpts[0];// ds[0] is q0!
    for(uint i=1;i<N;++i) ds[i]=cpts[i-1].conj()*cpts[i];
    for(uint i=0;i<N;++i) das[i]=ds[i].upow(a[i]);
    qs[0]=das[0];
    for(uint i=1;i<N;++i) qs[i] = qs[i-1]*das[i];
    return qs[N-1];
}
template <class T, int N>
inline Quaternion<T> compute_quaternion(
        const Vector<Vector<T,7>,N/*Degree+1*/>& state,
        const SplineBasisKoeffs& sbk)  {
    static_assert (N>=1," order is degree>0, +1" );
    return compute_qdot<T,N>(state, sbk.cbs<N-1>(0));
}



template<class T, uint N>
inline Pose<T> compute_pose(
        Vector<Vector<T,7>,N/*Degree+1*/> state,
        const Vector<double,N>& cbs)
{
    return Pose<T>(compute_quaternion<T,N>(state,cbs).q,
                   compute_translation<T,N>(state,cbs));
}

template<class T, uint N>
inline Pose<T> compute_pose(
        Vector<Vector<T,7>,N/*Degree+1*/> state,
        const SplineBasisKoeffs& sbk)
{
    return compute_pose<T,N>(state, sbk.cbs<N-1>(0));
}

template<class T, long unsigned int N>
inline Pose<T> compute_pose(const std::array<const T * const, N>& cpts,
                            const Vector<double,N>& cbs){
    return compute_pose(to_state(cpts),cbs);
}

template<class T> inline Vector3<T>
angular_velocity(const Quaternion<T>& q,
                 const Quaternion<T>& q_dt)
{
    // is this body? not sure...
    // the simple and safe version
    //auto w=(q_dt*q.conj())*T(2.0);return w.vec();
    // the faster and num better version
    return q_dt.imag_of_multiply_b_conj(q)*T(2.0);
}

template<class T> inline Vector3<T>
angular_velocity(const Vector<Quaternion<T>,3>& qs){
    return angular_velocity(qs[0],qs[1]);
}

template<class T, uint N> inline
Vector3<T> angular_velocity(
        const Vector<Vector<T,7>, N/*Degree+1*/>& state,
        const Vector3<Vector<double, N>>& as)
{
    Vector3<Quaternion<T>> qs=compute_qdot(state, as, true, false);
    return angular_velocity(qs[0],qs[1]);
}

template<class T, uint N> inline
Vector3<T> angular_velocity(
        const Vector<Vector<T,7>, N/*Degree+1*/>& state,
         const SplineBasisKoeffs& sbk)
{
    Vector3<Quaternion<T>> qs=compute_qdot(state, sbk.cbss<N-1>(), true, false);
    return angular_velocity(qs[0],qs[1]);
}

//////////////////////////// World variant /////////////////////////////

template<class T> inline Vector3<T>
angular_velocity_world(const Quaternion<T>& q,
                 const Quaternion<T>& q_dt)
{
    // am I sure which is which? NOT at all.. test
    return (q.conj()*q_dt).vec()*T(2.0); // faster variant exists...
}

template<class T> inline Vector3<T>
angular_velocity_world(const Vector<Quaternion<T>,3>& qs){
    return angular_velocity_world(qs[0],qs[1]);
}

template<class T, uint N> inline
Vector3<T> angular_velocity_world(
        const Vector<Vector<T,7>, N/*Degree+1*/>& state,
        const Vector3<Vector<double, N>>& as)
{
    Vector3<Quaternion<T>> qs=compute_qdot(state, as, true, false);
    return angular_velocity_world(qs[0],qs[1]);
}

template<class T, uint N> inline
Vector3<T> angular_velocity_world(
        const Vector<Vector<T,7>, N/*Degree+1*/>& state,
         const SplineBasisKoeffs& sbk)
{
    Vector3<Quaternion<T>> qs=compute_qdot(state, sbk.cbss<N-1>(), true, false);
    return angular_velocity_world(qs[0],qs[1]);
}
///



template<class T> inline Vector3<T>
angular_acceleration(const Vector<Quaternion<T>,3>& qs)
{
    // the nice and safe expression:
    //auto w=(qs[2]*qs[0].conj() + qs[1]*qs[1].conj())*T(2.0);        return w.vec();
    // hypothetically faster version, except its pretty much exactly as fast
    // fast and better num version...
    return qs[2].imag_of_multiply_b_conj(qs[0])*T(2.0);
}
template<class T, uint N> inline Vector3<T>
angular_acceleration(const Vector<Vector<T,7>, N/*Degree+1*/>& state,
                     const SplineBasisKoeffs& sbk){
return angular_acceleration(compute_qdot(state,sbk,true,true));
}
template<class T, uint N> inline Vector3<T>
angular_acceleration(const Vector<Vector<T,7>, N/*Degree+1*/>& state,
                     const Vector3<Vector<double, N>>& as){
return angular_acceleration(compute_qdot(state,as,true,true));
}



template<class T> inline Vector3<T>
angular_jerk(const Vector<Quaternion<T>,3>& qs)
{
    // the nice and safe expression:
    //auto w=(qs[2]*qs[0].conj() + qs[1]*qs[1].conj())*T(2.0);        return w.vec();
    // hypothetically faster version, except its pretty much exactly as fast
    // fast and better num version...
    return qs[2].imag_of_multiply_b_conj(qs[0])*T(2.0);
}
template<class T, uint N> inline Vector3<T>
angular_jerk(const Vector<Vector<T,7>, N/*Degree+1*/>& state,
                     const SplineBasisKoeffs& sbk){
return angular_jerk(compute_qdot(state,sbk,true,true));
}
template<class T, uint N> inline Vector3<T>
angular_jerk(const Vector<Vector<T,7>, N/*Degree+1*/>& state,
                     const Vector3<Vector<double, N>>& as){
return angular_jerk(compute_qdot(state,as,true,true));
}








template<class T> inline Vector3<T>
angular_derivative(
        const Vector<Quaternion<T>,3>& qs,
        int derivative){


    switch (derivative) {
    case 1: return angular_velocity(qs[0],qs[1]);
    case 2: return angular_acceleration(qs);
    default:{
        mlog()<<"invalid derivative"<<derivative<<"\n";
        return Vector3<T>::Zero();
    }
    }
}


}
