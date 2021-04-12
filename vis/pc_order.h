#pragma once
#include <vector>

#include <mlib/vis/order.h>
#include "mlib/utils/cvl/pose.h"
#include <mlib/utils/colormap.h>

namespace osg{class Node;}

namespace mlib{

class PC{
public:
    std::vector<cvl::Vector3d> xs;
    std::vector<Color> xs_cols;
    std::vector<cvl::PoseD> ps;
    std::vector<std::vector<cvl::PoseD>> posess;
    std::vector<Color> pose_colors;
    double coordinate_axis_length;
};
PC default_scene();

struct PCOrder:public Order{
    PCOrder(PC& pc, bool update=false):Order(update),pc(pc){}
    PC pc;
    osg::Node* group(double marker_scale) override;
};
}

