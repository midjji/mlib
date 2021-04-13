#pragma once
#include <mlib/datasets/kitti/odometry/kitti.h>
namespace cvl{
namespace kitti{
void convert2ng(KittiDataset kd,std::string outputpath);

}// end kitti namespace
}// end namespace cvl
