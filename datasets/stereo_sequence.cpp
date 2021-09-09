#include <mlib/datasets/stereo_sequence.h>
namespace cvl {

StereoSequence::~StereoSequence(){}
std::vector<PoseD> StereoSequence::gt_vehicle_poses() const{
    std::vector<PoseD> ps=gt_poses();
    // Should be Pwc*Pcv = Pwv
    for(auto& p:ps){        p=p*calibration().P_cam0_vehicle();    }
    return ps;
}

}

