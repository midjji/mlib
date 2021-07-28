#pragma once
#include <mlib/utils/cvl/pose.h>
namespace cvl{
namespace kitti{

/**
 * @brief The DistLsh class
 * KITTI specific distance hashing to speed up length quires
 */
class DistLsh{
public:
    DistLsh()=default;
    DistLsh(const std::vector<cvl::PoseD>& ps);
    /**
     * @brief getIndexPlusDist returns the index of the frame which is atleast dist after start but atmost dist+2m or -1 if no such frame is found
     * @param start
     * @param dist
     * @return
     */
    int getIndexPlusDist(uint start, double dist) const;
    explicit operator bool() const;
private:
    std::vector<double> dists;
    std::vector<std::vector<std::pair<uint,double>>> distmap;
};
}
}
