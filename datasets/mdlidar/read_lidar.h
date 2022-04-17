#pragma once
#include <vector>
#include <mlib/utils/cvl/pose.h>
#include <sstream>
#include <map>
namespace cvl {

namespace md {


struct LidarObservation
{

    using float128=long double;
    LidarObservation()=default;
    LidarObservation(float128 time,  float x, float y, float z, int frame_):time(time*1e-9)
    {
        xs[0]=x;
        xs[1]=y;
        xs[2]=z;
        frame=frame_;
    }
    float128 time; // in seconds;
    Vector3d xs; // in meter
    int frame;
};

std::vector<md::LidarObservation> read_lidar();
std::vector<md::LidarObservation> read_lidar_frame(int frame);

// frame delta time is 0.1
std::map<long double, cvl::PoseD> read_lidar_gt_poses();
// frame 2 pose
std::map<int, cvl::PoseD> read_lidar_gt_poses_by_frame();
}



}
std::string str(cvl::md::LidarObservation ob);
