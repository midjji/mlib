#pragma once
#include <mlib/datasets/frameid2time.h>
#include <vector>
namespace cvl{
namespace kitti
{
class Fid2Time :public Frameid2TimeMapLive
{
public:
    Fid2Time()=default;
    Fid2Time(const std::vector<double>& ts);

};

}
}


