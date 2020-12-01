#pragma once

#include <osg/Matrixd>
#include <mlib/utils/cvl/matrix.h>
namespace cvl{
osg::Matrixd cvl2osg(Matrix4d m);
osg::Vec3d cvl2osg(Vector3d v);
osg::Quat cvl2osgq(Vector4d q);
Vector3d osg2cvl(osg::Vec3d v);
Vector4d osg2cvl(osg::Quat q);

}// end namespace cvl


