#pragma once

#include <osg/Matrixd>
#include <mlib/utils/cvl/matrix.h>
#include <mlib/utils/cvl/pose.h>
namespace cvl{
osg::Matrixd cvl2osg(Matrix4d m);
osg::Vec3d cvl2osg(Vector3d v);
osg::Vec3f cvl2osgf(Vector3d v);
osg::Quat cvl2osgq(Vector4d q);
Vector3d osg2cvl(osg::Vec3d v);
Vector4d osg2cvl(osg::Quat q);
/** Convert an cvl 4x4 matrix (double) to a GL matrix */
osg::Matrixd cvl2osg(cvl::Matrix4d m);
osg::Matrixd cvl2osg(cvl::PoseD m);

}// end namespace cvl


