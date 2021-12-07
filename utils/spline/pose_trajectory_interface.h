#pragma once
#include <mlib/utils/cvl/pose.h>
#include <mlib/utils/spline/base.h>


namespace cvl {


struct PoseTrajectoryInterface:
        public BaseUniformSpline
{
    PoseTrajectoryInterface()=default;
    PoseTrajectoryInterface(double delta_time,
                            int degree,
                            TransformDirection direction);
    virtual ~PoseTrajectoryInterface();
    virtual std::vector<PoseD> display_poses(const std::vector<double>& ts) const;
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

private:
    TransformDirection direction_;


};



}
