#pragma once
#include <memory>
// fits int64 and uint64, this is what should have been used for std::chrono, if not a templated variable size
using float128 = long double;




namespace cvl{
class StereoSequence;
struct Sample
{
    enum {image, stereo, imu, multi_imu, lidar, hilti};


    Sample(float128 time, std::shared_ptr<StereoSequence> ss);
    virtual ~Sample();

    virtual float128 time() const;
    virtual int type() const=0;

    const std::shared_ptr<StereoSequence>sequence() const;

private:
    const float128 time_; // in seconds
    const std::shared_ptr<StereoSequence> wseq; // upgrade to shared and enable shared from this later...
};



}
