#include "triangulate.h"
#include "stereo_errors.h"
#include <ceres/problem.h>
#include <mlib/utils/mlog/log.h>
#include <mlib/utils/cvl/triangulate.h>
using std::cout;
using std::endl;
namespace cvl {
Vector3d triangulate(const StereoCalibration& intrinsics,
                     const std::vector<Vector2d>& obs,
                     const std::vector<PoseD>& ps,
                     Vector3d x,
                     bool reinitialize)
{
    std::vector<Vector3d> obss; obss.reserve(obs.size());
    for(const auto& ob:obs) obss.push_back(Vector3d(ob[0], ob[1], -1));

    return triangulate(intrinsics, obss, ps, x, reinitialize);
}
Vector3d triangulate(const StereoCalibration& intrinsics,
                     const std::vector<Vector3d>& obs,
                     const std::vector<PoseD>& ps,
                     Vector3d x,
                     bool reinitialize)
{
    /*
    cout<<"\nbegin triangulate:\n";
    for(int i=0;i<obs.size();++i)
    {
        auto yr=intrinsics.stereo_project(ps[i]*x);
        if(obs[i][2]<0)
            cout<<obs[i].drop_last()<<"\n"<<yr.drop_last()<<"\n";
        else
            cout<<obs[i]<<"\n"<<yr<<"\n";

    }*/


    require(obs.size()==ps.size()," must be the same size!");
    require(obs.size()>=2, "must have enough obs");

    if(reinitialize && false)
    {
        x=triangulate(ps[0], ps[1], intrinsics.undistort(obs[0].drop_last()), intrinsics.undistort(obs[1].drop_last()));
    }

    ceres::Problem problem;

    for(int i=0;i<obs.size();++i)
    {
       // std::cout<<ps[i]<<" "<<obs[i]<<std::endl;
        problem.AddResidualBlock(reprojection_cost<StereoCalibration, 3>(intrinsics, ps[i], obs[i]), nullptr, x.begin());
    }

    ceres::Solver::Options options;
    {
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.num_threads=1;// its so tiny its better to parralleize around many triangulations instead!
        options.max_num_iterations=50;
    }
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    /*
    std::cout<<summary.FullReport()<<std::endl;
    for(int i=0;i<obs.size();++i)
    {
        auto yr=intrinsics.stereo_project(ps[i]*x);
        if(obs[i][2]<0)
            cout<<obs[i].drop_last()<<"\n"<<yr.drop_last()<<"\n";
        else
            cout<<obs[i]<<"\n"<<yr<<"\n";
    }
    */


    return x;
}
}
