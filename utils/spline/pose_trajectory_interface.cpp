#include <mlib/utils/spline/pose_trajectory_interface.h>
#include <mlib/utils/cvl/integrate.h>

namespace cvl {
PoseTrajectoryInterface::PoseTrajectoryInterface(double delta_time, int degree,
                                                  TransformDirection direction):
    BaseUniformSpline(delta_time, degree),
    direction_(direction){}
PoseTrajectoryInterface::~PoseTrajectoryInterface(){}
std::vector<PoseD> PoseTrajectoryInterface::display_poses(const std::vector<double>& ts) const
{
    std::vector<PoseD> ps;ps.reserve(ts.size());
    // we assume its in x_world=P(t)x_camera, so invert for display? No, user must chose!
    for(auto t:ts)
        ps.push_back(this->operator()(t));
    return ps;
}
const TransformDirection PoseTrajectoryInterface::direction() const
{
    return direction_;
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
