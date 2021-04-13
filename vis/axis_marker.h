#pragma once
#include <mlib/utils/cvl/pose.h>

namespace osg {
class Node;
}
namespace mlib {


osg::Node* MakeAxisMarker(cvl::PoseD Pcw, float axis_length, float line_width);

osg::Node* MakeTrajectory(
        const std::vector<cvl::PoseD>& Pcws,
        float length=1,
        float width=1,
        float point_radius=0, // >0 means add colored ball at center
        cvl::Vector3d color=cvl::Vector3d(255,255,255));
}
