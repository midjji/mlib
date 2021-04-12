#pragma once
#include <mlib/utils/cvl/pose.h>

namespace osg {
class Node;
}
namespace vis {


osg::Node* MakeAxisMarker(cvl::PoseD pose, float axis_length, float line_width);
osg::Node* MakeTrajectory(const std::vector<cvl::PoseD>& poses, float length=2.0,float width=3.0);
}
