#pragma once
#include <vector>
#include <mlib/utils/cvl/matrix.h>
#include <mlib/utils/colormap.h>
namespace osg {
class Node;
}
namespace vis {
osg::Node* MakePointCloud(
        const std::vector<cvl::Vector3d>& points,
        const std::vector<mlib::Color>& colors,
        float radius);

osg::Node* MakePointCloud(
        const std::vector<cvl::Vector3d>& points,
        const std::vector<cvl::Vector3d>& colors,
        float radius);
}
