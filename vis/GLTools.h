#pragma once

#include <osg/Image>
#include <osgViewer/Viewer>
#include <opencv2/core/mat.hpp>

/*********************************************************************************************/
osg::ref_ptr<osg::Node> MakePointCloud(osg::ref_ptr<osg::Vec3Array> vertices, osg::ref_ptr<osg::Vec3Array> colors, float ptSize);
template<class PointType>

osg::ref_ptr<osg::Node> MakeGenericPointCloud(const std::vector<PointType>& points,
                                 float radius=2.0)
{

    osg::ref_ptr<osg::Vec3Array> xs = new osg::Vec3Array;xs->reserve(points.size());
    osg::ref_ptr<osg::Vec3Array> cs = new osg::Vec3Array;cs->reserve(points.size());
    for(uint i=0;i<points.size();++i) xs->push_back(osg::Vec3(points[i][0],points[i][1],points[i][2]));
    for(auto i=cs->size();i<points.size();++i) cs->push_back(osg::Vec3(0,0.5,0.5));
    osg::ref_ptr<osg::Node> pc=MakePointCloud(xs, cs, radius);
    return pc;
}
template<class PointType, class ColorType>
/**
 * @brief MakePointCloud
 * @param points
 * @param colors
 * @param radius
 * @return
 */
osg::ref_ptr<osg::Node> MakeGenericPointCloud(const std::vector<PointType>& points,
                                 const std::vector<ColorType>& colors,
                                 float radius=2.0)
{

    osg::ref_ptr<osg::Vec3Array> xs = new osg::Vec3Array;xs->reserve(points.size());
    osg::ref_ptr<osg::Vec3Array> cs = new osg::Vec3Array;cs->reserve(points.size());
    for(uint i=0;i<points.size();++i) xs->push_back(osg::Vec3(float(points[i][0]),float(points[i][1]),float(points[i][2])));
    for(uint i=0;i<colors.size();++i) cs->push_back(osg::Vec3(float(colors[i][0]/255.0),float(colors[i][1]/255.0),float(colors[i][2]/255.0)));
    for(uint i=cs->size();i<points.size();++i) cs->push_back(osg::Vec3(0,0.5,0.5));
    osg::ref_ptr<osg::Node> pc=MakePointCloud(xs, cs, radius);
    return pc;
}

/*********************************************************************************************/






osg::Geode *MakeImagePlane(osg::Image *img);
osgViewer::View *ConfigureOsgWindow(osgViewer::View *view, const std::string& name, int x, int y, unsigned int width, unsigned int height);
osgViewer::View *MakeOsgWindow(const std::string& name, int x, int y, unsigned int width, unsigned int height);
osg::MatrixTransform *MakeCameraIcon(osg::ref_ptr<osg::Image> photo, int width, int height,
                                     const osg::Matrix& pose, float flen, float f, const osg::Vec3& color);
osg::Node *MakeGrid(int num_squares, float side_length, const osg::Vec3& color, const osg::Matrix& pose);

osg::Node *MakeAxisMarker(float axis_length, float line_width, const osg::Matrix& pose);

osg::Geode *MakeAxisMarker(float axis_length, float line_width);
osg::MatrixTransform *MakeAxisMarker(const osg::Matrixd& pose, float axis_length, float line_width);


osg::Group* MakeTrajectory(const std::vector<osg::Matrixd>& poses, float length=2.0,float width=3.0);


void ConfigureOrthoCamera(int width, int height, osg::Camera* camera);

