#pragma once
#include <osg/Matrixd>
#include <osg/Image>

#include <opencv2/core.hpp>
#include <mlib/utils/cvl/pose.h>
#pragma GCC diagnostic push
#include <Eigen/Core>
#pragma GCC diagnostic pop
/** Convert an cvl 4x4 matrix (double) to a GL matrix */
osg::Matrixd CvlToGl(const cvl::Matrix4d& m);
osg::Matrixd CvlToGl(const cvl::PoseD& m);


/** Convert an Eigen 4x4 matrix (double) to a GL matrix */
osg::Matrixd CvToGl(const Eigen::Matrix4d& m);

/** Convert an Eigen 4x4 matrix (float) to a GL matrix */
osg::Matrixd CvToGl(const Eigen::Matrix4f& m);

/** Convert a GL matrix to an Eigen 4x4 matrix  */
Eigen::Matrix4d GlToCv(const osg::Matrixd& m);
/**
 * Create a GL projection matrix from camera intrinsics
 *
 * @note See http://strawlab.org/2011/11/05/augmented-reality-with-OpenGL/
 */
osg::Matrixd CvKToGlProjection(const cvl::Matrix3d& K,
                               float viewportWidth, float viewportHeight,
                               float zNear, float zFar);

osg::Matrixd CvPoseToGlPose(const Eigen::Matrix4f& cvPose);

/** Transform a Hartley-Zisserman camera pose to a GL view matrix. */
osg::Matrixd CvPoseToGlView(const Eigen::Matrix4d& cvPose);

/** Transform a Hartley-Zisserman camera pose to a GL view matrix. */
osg::Matrixd CvPoseToGlView(const Eigen::Matrix4f& cvPose);

/** Transform a GL view matrix to a Hartley-Zisserman camera pose. */
Eigen::Matrix4d GlViewToCvPose(const osg::Matrixd& glPose);

/** Generate a GL image from an OpenCV matrix */
osg::Image *CvMatToGlImage(const cv::Mat& src);








/** Generate an OpenCV matrix from a GL image.
 * Only 8-bit RGB supported right now. */
cv::Mat3b GlImageToCvMat(const osg::Image *src);
