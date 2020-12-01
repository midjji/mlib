#include <mlib/utils/smooth_trajectory.h>

namespace cvl{





std::vector<double> SmoothTrajectory::interior_times(int N, int border){
    // from 0 to 1/normf in N steps, so delta is
    std::vector<double> ts;ts.reserve(N);

    double d=(t1()-t0())/double(N);
    for(int i=border;i<N-border;++i){
     double time=t0()+d*i;
        ts.push_back(time);
    }
    return ts;
}
std::vector<PoseD> SmoothTrajectory::display_poses(int N, int border){
    std::vector<PoseD> ps;ps.reserve(N);
    auto ts=interior_times(N,border);

    for(auto t:ts)
        ps.push_back((*this)(t));

    return ps;
}
}
