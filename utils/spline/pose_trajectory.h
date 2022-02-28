#pragma once
#include <array>
#include <iostream>
#include <iomanip>


#include <mlib/utils/cvl/matrix.h>
#include <mlib/utils/mlog/log.h>


#include <mlib/utils/spline/coeffs.h>
#include <mlib/utils/spline/pose_trajectory_interface.h>
#include <mlib/utils/spline/pose_trajectory_helpers.h>

namespace cvl{








template<int Degree>
/**
 * @brief The PoseSpline class
 *
 * Rotation and translation spline with shared knots and order.
 *
 * \note
 * - it is better for imu observations and world position regularization
 * if the spline is x_world=Rx_sensor +t, but the choice is up to the user.
 * - inheriting spline could reduce code duplication, its not by much
 * and it may have performance penalties.
 * - decoupling the rotation and the translation is probably a good idea, but lets wait for now.
 *
 *
 *
 * Consider removing the templating on the degree to fix compile times.
 * Its required that it knows its own degree compile time to match the state lengths,
 * but using a bigger one would let me move almost all of it to a constexpr variable which is only used when interacting with ceres solver.
 * This in turn would make this allready complicated class more complicated... hmm going the other way would mean losing the perf
 * advantage of the constexpr formulation though,. No clear solution.
 */
class PoseSpline : public PoseTrajectoryInterface
{


public:
    static constexpr int Dims=7;
    static constexpr int degree(){return Degree;}
    static constexpr int order(){return Degree+1;}
    static constexpr int dims(){return 7;}
    static constexpr int local_window_size(){return order();}

    using ControlPoint_Ptr =ControlPoint_*;
    using ControlPointPtr =ControlPoint<7>*;


    // Does not take ownership!
    // call only once per problem for ideal speed!
    static auto local_parametrization(){

        // return nullptr if it lacks one...
        return new ceres::ProductParameterization(
                    new ceres::QuaternionParameterization(),
                    new ceres::IdentityParameterization(3));
    }

    PoseSpline()=default;
    PoseSpline(double delta_time, bool common_from):PoseTrajectoryInterface(delta_time, degree(), common_from){}


    //void set_time_range(double start_time, double end_time){}






    // generates on demand


    std::array<double*,Degree+1>
    view_control_point_params(double time)
    {
        // should this one insert? yes is best
        std::array<double*,Degree+1> arr;
        // this one must insert!
        int j=0;
        // unaffected by derivative, but the coefficients may be all zero always.
        for(int i=get_first(time);i<=get_last(time);++i)
            arr[j++]=control_point(i).begin();
        return arr;
    }
    Vector<Vector<double,Dims>,Degree+1>
    control_points_implied(double time) const{
        Vector<Vector<double,Dims>,Degree+1> arr;
        int j=0;
        // derivatives might decrease this in the real case.
        // but this is always all of them!
        for(int i=get_first(time);i<=get_last(time);++i){
            arr[j++]=control_point_implied(i);
        }
        return arr;
    }

    virtual Vector<double,Dims>
    control_point_delta_implied(int i, int delta) const
    {
        if(delta<1) mlog()<<"bad user\n";

        if(delta==1){
            if(i<=first_control_point())
                return control_point_delta_implied(first_control_point()+1,1);
            if(i>last_control_point())
                return control_point_delta_implied(last_control_point(),1);

            auto k0=control_point_implied(i);
            auto k1=control_point_implied(i-1);
            // what kind of delta should be used,?
            // geodesic makes sense, but I dont know that here...
            // euclidean for now
            return k0-k1;
        }
        if(delta ==2){
            if(i<first_control_point()+1){
                return Vector<double,Dims>::Zero();
            }
            if(i>last_control_point()+Degree){
                return Vector<double,Dims>::Zero();
            }
        }
        return control_point_delta_implied(i,delta-1) - control_point_delta_implied(i-1,delta-1);
    }


    PoseD pose(double time) const
    {
        SplineBasisKoeffs sbk=this->ccbs(time);
        return compute_pose<double,Degree+1>(
                    this->control_points_implied(time),
                    sbk.cbs<Degree>());
    }
    PoseD operator()(double time) const override{        return pose(time);    }
    std::vector<PoseD> operator()(std::vector<double> times, bool invert=false) const
    {
        std::vector<PoseD> ps;ps.reserve(times.size());
        for(double time:times) if(invert)
            ps.push_back(pose(time).inverse());
        else
            ps.push_back(pose(time));
        return ps;
    }

    // note translation derivatives lack the discontinuities, as they cannot be represented this way
    Vector3d translation(double time, int derivative) const override
    {
        return compute_translation( this->control_points_implied(time), ccbs(time), derivative);
    }

    Vector3d
    angular_velocity(double time) const override
    {
        return cvl::angular_velocity(this->control_points_implied(time),this->ccbs(time));
    }

    Vector3d
    angular_acceleration(double time) const override{
        return cvl::angular_acceleration(this->control_points_implied(time),this->ccbs(time));
    }


protected:

public:

    std::string serialize()
    {

        std::stringstream ss;
        ss<<std::setprecision(9);
        ss<<Degree<<" ";
        ss<<this->delta_time()<<" ";
        for(auto [i,cptwr]:this->control_points()){
            ss<<i<< " ";
            using ControlPointPtr=ControlPoint<Dims>*;
            auto& c=ControlPointPtr(cptwr.ptr)->x;
            for(auto d:c)
                ss<<d<<" ";
        }
        return ss.str();
    }

    static PoseSpline<Degree> deserialize(std::string str)
    {
        std::stringstream ss(str);
        int deg;
        ss>>deg;
        if(deg!=Degree) mlog()<<"deserializeing wrong degree!\n";
        double dt;
        ss>>dt;
        PoseSpline<Degree> p(dt,false);
        int i;
        while(ss>>i)
        {
            Vector4d q;
            for(int j=0;j<4;++j)
                ss>>q[j];
            q.normalize();
            Vector<double,7> s;
            for(int j=0;j<4;++j)
                s[j]=q[j];
            for(int j=4;j<7;++j)
                ss>>s[j];
            p.control_point(i)=s;

        }
        return p;
    }
};


double integrate_accelleration_squared(double t0, double t1, const PoseSpline<2>& ps);
double integrate_accelleration_squared(double t0, double t1, const PoseSpline<3>& ps);
double integrate_accelleration_squared(double t0, double t1, const PoseSpline<4>& ps);







} // end namespace cvl


template<unsigned int Order>
// order is probably one too high, check
std::ostream& operator<<(std::ostream& os,
                         cvl::PoseSpline<Order> spline){
    return os<<spline.display();
}
template<int Order>
// order is probably one too high, check
std::ostream& operator<<(std::ostream& os,
                         cvl::PoseSpline<Order> spline){
    return os<<spline.display();
}









