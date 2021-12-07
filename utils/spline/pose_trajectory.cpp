#include <mlib/utils/spline/pose_trajectory.h>

namespace cvl{


namespace  {
Vector3d tr(const Vector7d& x){return Vector3d(x[4],x[5],x[6]);}
}



double integrate_accelleration_squared(
        double t0, double t1,
        const PoseSpline<2>& ps)
{

    // accelleration is zero outside:
    auto ab = ps.integrate_accelleration_squared_time_cap(t0,t1);
    t0=std::get<0>(ab);    t1=std::get<1>(ab);

    if(t1<=t0) return 0;
    //cout<<"time: ("<<t0<< ", "<<t1<<"), ("<<t0p<< ", "<<t1p<<")"<<endl;
    double k=ps.delta_time();


    if(k!=1.0)
    {
        double a=t0/k;
        double b=t1/k;
        // eliminate delta_time by sub
        PoseSpline<2> cardinal=ps;;
        cardinal.set_delta_time(1);
        return std::pow(k,-3.0)*integrate_accelleration_squared(a,b,cardinal);
    }

    auto mn = ps.integrate_accelleration_squared_cpts_needed(t0,t1);    int M=std::get<0>(mn);    int N=std::get<1>(mn);

    auto delta_alpha=[&](int i, int delta) { return tr(ps.control_point_delta_implied(i,delta));    };

    double boundrary_term=
            ps.translation(t1,1).dot(ps.translation(t1,2)) -
            ps.translation(t0,1).dot(ps.translation(t0,2));
    double interior_term = 0;
    for(int i=M;i<=N;++i)
    {
        double it=ps.translation(i,1).dot(delta_alpha(i,3));
        interior_term+=it;
    }
    return boundrary_term - interior_term;
}
double integrate_accelleration_squared(double t0, double t1, const PoseSpline<3>& ps)
{

    // accelleration is zero outside:
    auto ab = ps.integrate_accelleration_squared_time_cap(t0,t1);    t0=std::get<0>(ab);    t1=std::get<1>(ab);

    if(t1<=t0) return 0;
    double k=ps.delta_time();

    if(k!=1.0)
    {
        double a=t0/k;
        double b=t1/k;
        // eliminate delta_time by sub
        PoseSpline<3> cardinal=ps;
        cardinal.set_delta_time(1);

        return std::pow(k,-3.0)*integrate_accelleration_squared(a,b,cardinal);
    }
    auto mn = ps.integrate_accelleration_squared_cpts_needed(t0,t1);    int M=std::get<0>(mn);    int N=std::get<1>(mn);

    auto delta_alpha=[&](int i, int delta)
    {        return tr(ps.control_point_delta_implied(i,delta));
    };



    // \int_a^b s''^2 dt =
    double boundrary_term  = ps.translation(t1,1).dot(ps.translation(t1,2)) - ps.translation(t0,1).dot(ps.translation(t0,2));
    double boundrary_term2 = ps.translation(t1,0).dot(ps.translation(t1,3)) - ps.translation(t0,0).dot(ps.translation(t0,3));
    boundrary_term -= boundrary_term2;
    //cout<<"boundrary_term:    "<<boundrary_term<<endl;
    //\sum_M^N s'(i)\Delta_2 \gamma_i
    double interior_term = 0;
    for(int i=M;i<=N;++i){
        double it=ps.translation(i,0).dot(delta_alpha(i,4));
        //cout<<"it:                "<<it<<endl;
        interior_term+=it;
    }
    //cout<<"interior terms:    "<<interior_term<<endl;
    return boundrary_term + interior_term;
}
double integrate_accelleration_squared(double t0, double t1, const PoseSpline<4>& ps)
{
    // accelleration is zero outside:
    auto ab = ps.integrate_accelleration_squared_time_cap(t0,t1);    t0=std::get<0>(ab);    t1=std::get<1>(ab);

    if(t1<=t0) return 0;
    double k=ps.delta_time();

    if(k!=1.0)
    {
        double a=t0/k;
        double b=t1/k;
        // eliminate delta_time by sub
        PoseSpline<4> cardinal=ps;
        cardinal.set_delta_time(1);
        return std::pow(k,-3.0)*integrate_accelleration_squared(a,b,cardinal);
    }
    auto mn = ps.integrate_accelleration_squared_cpts_needed(t0,t1);    int M=std::get<0>(mn);    int N=std::get<1>(mn);
    auto delta_alpha=[&](int i, int delta)
    {        return tr(ps.control_point_delta_implied(i,delta));
    };



    double boundrary_term0  = ps.translation(t1,1).dot(ps.translation(t1,2)) - ps.translation(t0,1).dot(ps.translation(t0,2));
    double boundrary_term1 = ps.translation(t1,0).dot(ps.translation(t1,3)) - ps.translation(t0,0).dot(ps.translation(t0,3));
    double boundrary_term2 = ps.translation(t1,-1).dot(ps.translation(t1,4)) - ps.translation(t0,-1).dot(ps.translation(t0,4));
    double boundrary = boundrary_term0 - boundrary_term1 + boundrary_term2;

    double interior_term = 0;
    for(int i=M;i<=N;++i){
        double it=ps.translation(i,-1).dot(delta_alpha(i,5));
        //cout<<"it:                "<<it<<endl;
        interior_term+=it;
    }
    //cout<<"interior terms:    "<<interior_term<<endl;
    return boundrary - interior_term;

}

}
