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
    double coordinate_axis_length=1;
void fill_colors();
};
PC default_scene();

struct PCOrder:public Order{
    PCOrder(const PC& pc, bool clear_scene=true);
    PC pc;
    double scale=1;
    osg::Node* group() override;
};


struct PointsOrder:public Order{


    PointsOrder(const std::vector<cvl::Vector3d>& xs,
                Color color=Color::green(),
                bool clear_scene=true, double radius=1);
    PointsOrder(const std::vector<cvl::Vector3d>& xs,
                const std::vector<cvl::Vector3d>& colors, // rgb 0-255
                bool clear_scene=true, double radius=1);
    PointsOrder(const std::vector<cvl::Vector3d>& xs,
                const std::vector<Color>& colors,
                bool clear_scene=true, double radius=1);
    osg::Node* group() override;
private:
    std::vector<cvl::Vector3d> xs;
    std::vector<cvl::Vector3d> cs;
    double radius;
};


}

