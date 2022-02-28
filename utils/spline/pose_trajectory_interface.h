#pragma once
#include <mlib/utils/cvl/pose.h>
#include <mlib/utils/spline/base.h>


namespace cvl {


struct PoseTrajectoryInterface:
        public BaseUniformSpline
{
private:
    bool common_from_=false; // Pwa
public:
    bool common_from() const{return common_from_;}
    using ControlPoint_Ptr =ControlPoint_*;
    using ControlPointPtr =ControlPoint<7>*;
    PoseTrajectoryInterface()=default;
    PoseTrajectoryInterface(double delta_time,
                            int degree,
                            bool common_from);
    virtual ~PoseTrajectoryInterface();
    virtual std::vector<PoseD> display_poses(const std::vector<double>& ts, bool invert=true) const;

    virtual Vector3d angular_velocity(double time) const=0;
    virtual Vector3d angular_acceleration(double time) const=0;
    virtual Vector3d translation(double time, int derivative) const=0;
    virtual PoseD operator()(double time) const=0;
    const TransformDirection direction() const;
    void normalize_qs();
    double integrate_translation_derivative_squared_approximate(double t0,
                                                                double t1,
                                                                int derivative,
                                                                int num=1e5) const;


    ControlPoint_Ptr make_control_point(int i) const override;
    Vector7d initialize_impl(int i) const;
    Vector<double,7>& control_point(int i);
    Vector<double,7> control_point_implied(int i) const;
    void set_control_points_at_time(double time, PoseD P);
    std::vector<double*> view_control_point_params_dynamic(double time);
    std::vector<Vector<double,7>> control_points_implied_dynamic(double time) const;


    template<int Degree>
    Vector<Vector<double,7>,Degree+1>
    control_points_implied_for_degree(double time) const
    {
        Vector<Vector<double,7>,Degree+1> arr;
        int j=0;
        // derivatives might decrease this in the real case.
        // but this is always all of them!
        for(int i=get_first(time);i<=get_last(time);++i){
            arr[j++]=control_point_implied(i);
        }
        return arr;
    }


private:
    TransformDirection direction_;


};

}
