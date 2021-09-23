#include <mlib/datasets/imu_sample.h>

namespace cvl{
ImuSample::ImuSample(imu::Data data,const std::shared_ptr<StereoSequence> ss):
Sample(data.time,ss), data(data){}
int ImuSample::type() const{return imu;}



}
