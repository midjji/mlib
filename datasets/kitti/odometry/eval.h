#pragma once
#include <mlib/datasets/kitti/odometry/result.h>

namespace cvl{
namespace kitti{


std::map<int,Result> evaluate(KittiDataset& kd,
                              std::string estimatepath,  /** @param estimatepath path to the estimation output directories */
                              std::string estimate_name,  /** estimate names */
                              std::string outputpath    /** @param outputpath   where to store the resulting files */);

std::map<int,Result> evaluate(KittiDataset& kd,
                              // sequence to estimate poses
                              std::map<int,std::vector<PoseD>> Pwcs,
                              std::string estimate_name,
                              std::string outputpath    );

Result evaluate(Sequence& kd,
                std::vector<PoseD> Pwcs,
                std::string estimate_name,
                std::string outputpath    );

Result evaluate(Sequence& seq, const std::vector<PoseD>& Pwcs);



}// end kitti namespace
}// end namespace cvl
