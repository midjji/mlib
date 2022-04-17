#include <mlib/datasets/mdlidar/read_lidar.h>

#include <mlib/vis/mlib_simple_point_cloud_viewer.h>
#include <mlib/utils/mlibtime.h>
#include <iostream>
#include <mlib/utils/spline/pose_trajectory.h>
using namespace std;

namespace cvl {


void show_basic(){
    for(int i=0;i<3000;++i)
    {

        std::vector<cvl::md::LidarObservation> obs=cvl::md::read_lidar_frame(i);
        std::vector<cvl::Vector3d> xs;xs.reserve(obs.size());
        std::vector<mlib::Color> cs;cs.reserve(obs.size());

        for(auto& ob:obs)
        {
            xs.push_back(ob.xs*0.1);
            cs.push_back(mlib::Color::cyan());
        }

        mlib::pc_viewer("md basic lidar points")->setPointCloud(xs,cs);

     break;
    }
}


void show_offset()
{
  std::map<int, PoseD> poses=cvl::md::read_lidar_gt_poses_by_frame();
    for(int i=0;i<3000;++i)
    {
        PoseD Pwv=poses[i];

   mlib::sleep(0.1);
        std::vector<cvl::md::LidarObservation> obs=cvl::md::read_lidar_frame(i);
        std::vector<cvl::Vector3d> xs;xs.reserve(obs.size());
        std::vector<mlib::Color> cs;cs.reserve(obs.size());

        for(auto& ob:obs)
        {
            auto x=Pwv*ob.xs;
            xs.push_back(x*0.1);
            cs.push_back(mlib::Color::nr(ob.frame));
        }


        mlib::pc_viewer("md offset lidar points")->setPointCloud(xs,cs);

break;
    }
}

PoseSpline<1> trajectory()
{
    PoseSpline<1> ps(0.1,true);
    std::map<long double, PoseD> obs=cvl::md::read_lidar_gt_poses();
    for(auto [time, pose]:obs)
    {
        ps.set_control_points_at_time(time, pose);
    }
    return ps;
}
void show_rectified()
{
    auto Pvi=trajectory();
    for(int i=0;i<3000;++i)
    {

        mlib::sleep(3);
        std::vector<cvl::md::LidarObservation> obs=cvl::md::read_lidar_frame(i);
        std::vector<cvl::Vector3d> xs;xs.reserve(obs.size());
        std::vector<mlib::Color> cs;cs.reserve(obs.size());

        for(auto& ob:obs)
        {
            auto x=Pvi(ob.time)*ob.xs;
            xs.push_back(x*0.1);
            cs.push_back(mlib::Color::nr(ob.frame));
        }


        mlib::pc_viewer("md rectified lidar points")->setPointCloud(xs,cs);

break;
    }
}
void show_trajectory()
{

    std::map<long double, PoseD> obs=cvl::md::read_lidar_gt_poses();
    std::vector<PoseD> ps;ps.reserve(obs.size());
    std::vector<PoseD> pis;pis.reserve(obs.size());
    for(auto [time, pose]:obs)
    {
        // viewer expects Pcw;
        ps.push_back(pose);
        pis.push_back(pose.inverse());
    }
    mlib::pc_viewer("md lidar trajectory")->setPointCloud(ps);
    mlib::pc_viewer("md lidar trajectory Piv")->setPointCloud(pis);

}

}
int main()
{
    //cvl::show_trajectory();
    //cvl::show_basic();
    cvl::show_offset();
    cvl::show_rectified();
    mlib::sleep(10000);
    mlib::wait_for_viewers();
    return 0;
}
