#include <mlib/datasets/kitti/odometry/fid2time.h>
#include <sstream>
namespace cvl{
namespace kitti{

Fid2Time::Fid2Time(const std::vector<double>& ts) {
    for(int i=0;i<int(ts.size());++i)
        add(i,ts[i]);
}


}
}

