#pragma once
#include <mlib/datasets/sample.h>
#include <mlib/utils/imu_data.h>


namespace cvl{

struct ImuSample: public Sample
{
    imu::Data data;
    ImuSample(imu::Data data,StereoSequence* ss);
    virtual ~ImuSample();
    virtual int type() const override;
};


}
