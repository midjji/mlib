#include <mlib/utils/spline/pose_trajectory_interface.h>
#include <mlib/utils/cvl/integrate.h>

namespace cvl {
PoseTrajectoryInterface::PoseTrajectoryInterface(double delta_time, int degree,
                                                  bool common_from_):
    BaseUniformSpline(delta_time, degree),
    common_from_(common_from_){}
PoseTrajectoryInterface::~PoseTrajectoryInterface(){}
std::vector<PoseD> PoseTrajectoryInterface::display_poses(const std::vector<double>& ts, bool invert) const
{
    std::vector<PoseD> ps;ps.reserve(ts.size());
    // we assume its in x_world=P(t)x_camera, so invert for display? No, user must chose!
    for(auto t:ts)
        ps.push_back(this->operator()(t));
    if(invert){
        for(auto& p:ps)
            p=p.inverse();
    }
    return ps;
}

void PoseTrajectoryInterface::normalize_qs(){
    for(const auto& [i,s]:control_points())
    {
        double* ptr=s.begin();
        PoseD p(ptr, true);
        if(std::abs(p.q().norm()-1)>1e-9)
            mlog()<<"forgot to set local param"<<"\n";
        p.normalize();

        for(int i=0;i<7;++i)
        {
        ptr[i]=p[i];
        }
    }
}
std::vector<double*>
PoseTrajectoryInterface::view_control_point_params_dynamic(double time)
{
    std::vector<double*> arr;arr.reserve(degree()+1);

    for(int i=get_first(time);i<=get_last(time);++i)
        arr.push_back(control_point(i).begin());
    return arr;
}
std::vector<Vector<double,7>> PoseTrajectoryInterface::control_points_implied_dynamic(double time) const
{
    std::vector<Vector<double,7>> arr;arr.reserve(degree()+1);

    for(int i=get_first(time);i<=get_last(time);++i)
        arr.push_back(control_point_implied(i));
    return arr;
}
void PoseTrajectoryInterface::set_control_points_at_time(double time, PoseD P){
    std::vector<double*> arr=view_control_point_params_dynamic(time);
    for(double* ptr:arr)
        for(int i=0;i<7;++i)
            ptr[i] = P.qt()[i];
}

PoseTrajectoryInterface::ControlPoint_Ptr PoseTrajectoryInterface::make_control_point(int i) const{        return new ControlPoint<7>(initialize_impl(i));    }
Vector7d PoseTrajectoryInterface::initialize_impl(int i) const {

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


Vector<double,7>& PoseTrajectoryInterface::control_point(int i) {
    return ControlPointPtr(control_point_ptr(i))->x;
}
Vector<double,7> PoseTrajectoryInterface::control_point_implied(int i) const
{
    ControlPoint_Ptr ptr=get(i);
    if(ptr)
        return ControlPointPtr(ptr)->x;
    return initialize_impl(i);
}











double PoseTrajectoryInterface::integrate_translation_derivative_squared_approximate(double t0,
                                                            double t1,
                                                            int derivative,
                                                            int num) const
{
    auto tmp=[&](double t)
    {
        auto tr=translation(t, derivative);
        return tr.dot(tr);
    };
    return integrate_fast_mp(tmp,t0,t1,num);
}


}
