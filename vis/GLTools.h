#if 0
#pragma once

#include <osg/Image>
#include <osgViewer/Viewer>
#include <opencv2/core/mat.hpp>




osg::Geode *MakeImagePlane(osg::Image *img);
osgViewer::View *ConfigureOsgWindow(osgViewer::View *view, const std::string& name, int x, int y, unsigned int width, unsigned int height);
osgViewer::View *MakeOsgWindow(const std::string& name, int x, int y, unsigned int width, unsigned int height);
osg::MatrixTransform *MakeCameraIcon(osg::ref_ptr<osg::Image> photo, int width, int height,
                                     const osg::Matrix& pose, float flen, float f, const osg::Vec3& color);
osg::Node *MakeGrid(int num_squares, float side_length, const osg::Vec3& color, const osg::Matrix& pose);






void ConfigureOrthoCamera(int width, int height, osg::Camera* camera);

#endif
