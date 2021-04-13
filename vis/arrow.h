#pragma once
#include <mlib/utils/cvl/matrix.h>
#include <mlib/vis/flow.h>
namespace osg {
class Node;
}
namespace vis {
osg::Node* create_arrow(cvl::Vector3d from, cvl::Vector3d to, cvl::Vector3d color);
osg::Node* create_arrow(cvl::Flow flow);

}
