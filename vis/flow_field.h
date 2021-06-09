#pragma once
#include <vector>
#include <memory>

#include <mlib/vis/order.h>
#include "mlib/utils/cvl/pose.h"
#include <mlib/utils/colormap.h>
#include <mlib/vis/flow.h>


namespace osg{class Node;}

namespace mlib {



class Trajectory{
public:
    Trajectory()=default;
    Trajectory(std::vector<cvl::PoseD> poses);
    std::vector<cvl::PoseD> poses; // tramsforms from world to camera
    // additional camera information?
    void apply_transform(cvl::PoseD pose);
    bool is_normal() const;
};

struct PointCloud{
    std::vector<cvl::Vector3d> points, colors;

    PointCloud()=default;
    PointCloud(std::vector<cvl::Vector3d> points,
               std::vector<cvl::Vector3d> colors);

    void apply_transform(cvl::PoseD pose);
    void clean();
    void append(PointCloud pc);
};
class FlowField{
public:
    FlowField()=default;
    FlowField(std::vector<cvl::Flow> flows,
              PointCloud points,
              std::vector<Trajectory> trajectories);
    void apply_transform(cvl::PoseD pose);
    void clean();
    void cap_velocity(double len);
    void append(std::shared_ptr<FlowField> ff);

    std::vector<cvl::Flow> flows;
    PointCloud points;
    std::vector<Trajectory> trajectories;
};

struct FlowOrder:public mlib::Order{
    FlowOrder(const FlowField& ff, bool update=false);
    FlowField ff;
    double scale=1;
    osg::Node* group() override;
};


osg::Node* createFlowField(const FlowField& ff, double marker_scale);




}
