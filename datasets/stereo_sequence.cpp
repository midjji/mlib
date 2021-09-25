#include <mlib/datasets/stereo_sequence.h>
namespace cvl {

StereoSequence::~StereoSequence(){}
std::vector<PoseD> StereoSequence::gt_vehicle_poses() const{
    std::vector<PoseD> ps=gt_poses();
    // Should be Pwc*Pcv = Pwv
    for(auto& p:ps){        p=p*calibration().P_left_vehicle();    }
    return ps;
}
int StereoSequence::id() const{return 0;}
std::vector<PoseD> StereoSequence::gt_poses() const{
    return std::vector<PoseD>();
}
double StereoSequence::framerate() const{
    return 10;
}

}

