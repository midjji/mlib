#pragma once
#include <memory>
#include <vector>

#include <mlib/datasets/stereo_sequence.h>

namespace cvl
{

struct StereoDataset
{
    using sample_type=typename std::shared_ptr<StereoSample>;
    ~StereoDataset();
    virtual std::vector<std::shared_ptr<StereoSequence>> sequences() const=0;
};

}

