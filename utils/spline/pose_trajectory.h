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
    PoseSpline(double delta_time, TransformDirection direction):PoseTrajectoryInterface(delta_time, degree(), direction){}


    //void set_time_range(double start_time, double end_time){}
    void set_control_points_at_time(double time, PoseD P){
        std::array<double*,Degree+1> arr=this->view_control_point_params(time);
        for(double* ptr:arr)
            for(int i=0;i<7;++i)
                ptr[i] = P.qt()[i];
    }





    // generates on demand
    Vector<double,Dims>& control_point(int i) {
        return ControlPointPtr(control_point_ptr(i))->x;
    }
    Vector<double,Dims> control_point_implied(int i) const
    {
        ControlPoint_Ptr ptr=get(i);
        if(ptr)
            return ControlPointPtr(ptr)->x;
        return initialize_impl(i);
    }

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


    PoseD pose(double time) const{
        SplineBasisKoeffs sbk=this->ccbs(time);
        return compute_pose<double,Degree+1>(
                    this->control_points_implied(time),
                    sbk.cbs<Degree>());
    }
    PoseD operator()(double time) const override{        return pose(time);    }

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
    ControlPoint_Ptr make_control_point(int i) const override{
        return new ControlPoint<7>(initialize_impl(i));
    }
    virtual Vector7d initialize_impl(int i) const {

        /**
         * @brief The PoseControlPointIniter struct
         *
         * Predicts an existing control point as it self.
         * Predicts the first control point as identity,
         * Predicts control points before the first one as the first one.
         * Predicts after last as last.
         * Predicts between two control points using linear interpolation/slerp. This is almost certainly always the right choice.
         *
         * The specific goal of the initer is to select the values which maximize the chance
         * that they are in the convergence basin, not the values closest to the gt.
         *
         *
         *
         * For a discussion of why I chose as i did above, see below.
         *
         * If you think you can do better, note that changing the first point into some other constant
         * is reasonable. but you dont need to use this struct for that.
         * Changing the extrapolation before and after to constant omega has formulas below, todo, make it a option.
         * But the edges should be extrapolated by regularizers, not this way.
         * If you want a higher order slerp, just dont bother,
         * the optimizer will do it for you instead.
         *
         * Initializing the spline is more difficult than one might assume.
         *
         * Note, in general the spline needs be initialized incrementally,
         * i.e. start with a small connected subset and use a regularizer to extrapolate edges,
         * then add more observations connected to the set and so on.
         * This generally works well, but is slow. and the set might have to be very small.
         * Especially since adding regularizers generally makes the convergence basin smaller
         * as the spline can no longer optimize by rolling around the maximum rotation speed of
         * the delta and get back to speed zero.
         * This is a problem since imu measurements behave the same way for bad inits.
         *
         * The problematic case we ran into was that of estimating the motion of a camera
         * we had poses at certain times, but not often enough to make the spline fully connected.
         * Thus the two part process of estimating the spline using just these poses
         * then adding imu, appeared to work in step one, but the
         * poor init caused by the nearest neighbour control point init in the gaps
         * between observed poses caused failure.
         * Basically they all returned to origin there, which was difficult to debug.
         * Even when changed to use the nearest controlpoint, the outcome was still failures,
         * due to the meeting point between such extrapolations.
         *
         *
         * A simple solution would be to add a very weak first order
         * gen pspline geodesic regularizer when estimating,
         * connecting each sequential pair.
         * Relatively cheap and it will fill in from the first to the last,
         * and can be made weak enought it has almost no effect.
         *
         * This approach works, but the real problem is that spline does not init
         * in a reasonable way.
         *
         *
         * So what would be more reasonable?
         *
         * Extrapolation outside the control point span can be done using either
         * the asymptotic omega assumption or nearest neighbour.
         * In truth odds are both are rather bad,
         * the former causes guaranteed failures if it predicts more than a
         * full turn compared to the truth.
         * The latter is rather restrictive.
         *
         * Next there is the problem of values between known parts.
         * Here the answer is simpler. constant init and nearest neighbour
         *  is almost guaranteed to cause a knot where the change is terribly fast
         *   which always has worse or the same convergence basin as the slowest possible change
         *   between the endpoints.
         *
         *   Potentially we could still do better, e.g. using the derivatives at the edges,
         *   but that should be a user choice, and its a dumb one. A model strong enough to predict multiple loops which is both right, does so, and has good enough inputs to get it right, is very difficult to get. And if the alternative converges without having this, so would the slerp.
         *
         *   Thus, by default when asked what the control point between to known ones are,
         *   We provide the slerp between the two control points.
         *
         * We have three cases:
         * Before beginning, after end, and inbetween two known.
         * Note: if Spline is in extrapolation mode, we will never be asked for a point before or after, only inbetween, and that last case is unaffected by mode.
         *
         * Method 1: The Nearest neighbour is easy but leads to sharp changes in the derivatives, think gibbs phenomena but for spline basis.
         *
         * Method 2: We can use nearby control point differences to predict.
         *
         * For before beginning and after end, we could assume converging towards constant omega model.
         * This leads to:
         *      begin and end can be computed K_a^cK_{b+1} = K_{a-1}^cK_a => K_{a-1}, K_{b+1}
         *      if we want far after end, K_{e+N} = w^Nb where w=K_{e-1}K_e, N>=0
         *      if we want before K_{b-N} = K_{b}w^{-N} N>=0,
         * However, its unclear if this is better or worse than just nearest neighbour.
         *
         * The intermediate values i.e. K_a, missing, ..., missing, K_b, we have several choices.
         * * This method is better than nearest, but the sharp decline is still a problem!
         *
         *
         *
         *
         * // fundamentally extrapolating this way means we go towards a constant omega,
         *  dominated by the last measurements.
         *  But it does not mean we keep the omega we had at the last measurements,
         *  unless that was constant for some time allready.
         *
         * The interpolation
         *
         * Note,
         * - If a spline has disconnected elements when all costs have been added,
         *  this is not easy to see in general,
         * but if it is outright lacking some elements that should probably be
         * adressed by adding a very weak Gen-Pspline penalty to the missing ones.
         * Actually, for user safety, it seems likely a good idea to always add a
         * weak GP penalty along the entire spline.
         * But it should be optional. However, as this should be practice, we can assume that
         * the interpolation of missing values only needs to be decent, not perfect.
         */
        if(this->empty())
            return {1.0,0,0,0,0,0,0};

        if(i<this->control_points().cbegin()->first)
            return ControlPointPtr(this->control_points().cbegin()->second.ptr)->x;

        // or after
        if(i>this->control_points().crbegin()->first)
            return ControlPointPtr(this->control_points().crbegin()->second.ptr)->x;

        // exists, calling this way indicates a api missuse
        {
            auto it=this->control_points().find(i);
            if(it!=this->control_points().end()){
                mlog()<<"trying to init known keypoint: "<<i<<"\n";
                return ControlPointPtr(it->second.ptr)->x;
            }
        }

        // there can be holes in the sequence...,
        // get the preceeding point
        auto it=this->control_points().lower_bound(i); // gives first element >= i // really really dumb name...

        // prev and next exists, also its not key, due to tests above...
        auto prev=it; prev--;
        auto next=it;
        double fraction = double(i - next->first)/double(next->first - prev->first);





        PoseD Pprev=PoseD(ControlPointPtr(prev->second.ptr)->x);
        PoseD Pnext=PoseD(ControlPointPtr(next->second.ptr)->x);
        PoseD Pinterp=cvl::interpolate(Pprev, Pnext,fraction).qt();
        //mlog()<<"warning interpolation is untested i:"<<i<<" and "<<prev->first<<" to "<<next->first<<"\n";
        //std::cout<<Pprev<<"\n"<<Pnext<<"\n"<<Pinterp<<std::endl;
        return Pinterp.qt();

    }
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

    static PoseSpline<Degree> deserialize(std::string str, TransformDirection direction)
    {
        std::stringstream ss(str);
        int deg;
        ss>>deg;
        if(deg!=Degree) mlog()<<"deserializeing wrong degree!\n";
        double dt;
        ss>>dt;
        PoseSpline<Degree> p(dt, direction);
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









