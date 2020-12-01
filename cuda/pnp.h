#if 0
// this really! does not work...

#pragma once
#include <mlib/cuda/cuda_helpers.h>
#include <mlib/utils/cvl/pose.h>
//#include <mlib/sfm/p3p/klas.pnpd.h>
#include <mlib/utils/mlibtime.h>
#include <mlib/cuda/devmemmanager.h>
#include <mlib/sfm/p3p/parameters.h>

namespace cvl {
/**
 * \namespace cvl::cuda
 * \brief cvl::cuda Contains cvl things that require cuda
 *
 */
namespace cuda {
/**
 * @brief The PNPC class
 * assumes a second camera on the right at baseline distance.
 * yns=P*x_w
 * xr=Prl*P*x_w
 */
class PNPC{
public:


    PNPC(mlib::PnpParams params){
        this->params=params;
    }
    // replaces
    /**
     * @brief init
     * @param xs
     * @param yns pinhole normalized coordinates
     * @param disps pinhole normalized coordinates >0 pixel disp/K(0,0)
     * @param baseline
     */
    void init(const std::vector<cvl::Vector3d>& xs,
              const std::vector<cvl::Vector2d>& yns,
              const std::vector<double>& disps,
              double baseline);

    void compute();
    int getIters(){return dev_poses.cols;}
    PoseD getSolution(){return bestPose;}

private:
    PoseD bestPose;
    DevMemManager dmm; // may not be copied!
    int N=250;
    MatrixAdapter<PoseD>    dev_poses;
    MatrixAdapter<Vector3d> dev_xs;
    MatrixAdapter<Vector2d> dev_yns;
    MatrixAdapter<double>   dev_disps;
    MatrixAdapter<int>      dev_inlierss;


    mlib::PnpParams params;
    std::vector<Vector3d> xs;
    std::vector<Vector2d> yns;
    std::vector<double> disps;
    double baseline, thr;
};

}
}

#endif
