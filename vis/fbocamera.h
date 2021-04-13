#if 0
#pragma once

#include <mlib/utils/cvl/matrix.h>
#include <osg/MatrixTransform>
#include <osgViewer/Viewer>
#include <osg/Geode>
#include <osg/LineWidth>
#include <osg/Point>
#include <osg/Texture2D>



struct FBOCamera
{
	osg::ref_ptr<osg::Texture2D> texture;
	osg::ref_ptr<osg::FrameBufferObject> fbo;
	osg::ref_ptr<osg::Camera> osgCamera;

    FBOCamera(const int width, const int height, const cvl::Matrix3d& K);
	osg::Node *CreateQuad(const float width, const float height);
};



#endif
