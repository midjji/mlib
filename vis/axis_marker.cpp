#include <osg/Geode>
#include <osg/Geometry>
#include <osg/MatrixTransform>
#include <osg/LineWidth>

#include <osgGA/StateSetManipulator>

#include <mlib/vis/axis_marker.h>
#include <mlib/vis/point_cloud.h>
#include <mlib/vis/convertosg.h>

namespace mlib {



osg::Node* MakeAxisMarker(cvl::PoseD p, float axis_length, float line_width)
{

    osg::Matrixd pose=cvl2osg(p.inverse());
    osg::Vec3Array *vertices = new osg::Vec3Array;
    vertices->push_back(osg::Vec3(0,0,0));
    vertices->push_back(osg::Vec3(axis_length,0,0));
    vertices->push_back(osg::Vec3(0,0,0));
    vertices->push_back(osg::Vec3(0,axis_length,0));
    vertices->push_back(osg::Vec3(0,0,0));
    vertices->push_back(osg::Vec3(0,0,axis_length));

    osg::Vec3Array *colors = new osg::Vec3Array;
    colors->push_back(osg::Vec3(1,0,0));
    colors->push_back(osg::Vec3(0,1,0));
    colors->push_back(osg::Vec3(0,0,1));

    osg::Geometry *lines = new osg::Geometry;
    lines->setVertexArray(vertices);
    lines->setColorArray(colors, osg::Array::BIND_PER_PRIMITIVE_SET);
    lines->addPrimitiveSet(new osg::DrawArrays(GL_LINES, 0, 2));
    lines->addPrimitiveSet(new osg::DrawArrays(GL_LINES, 2, 2));
    lines->addPrimitiveSet(new osg::DrawArrays(GL_LINES, 4, 2));

    osg::Geode *marker_geode = new osg::Geode;
    marker_geode->addDrawable(lines);

    osg::PolygonMode *pm = new osg::PolygonMode(osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::LINE);
    osg::LineWidth *lw = new osg::LineWidth(line_width);

    osg::StateSet *state = marker_geode->getOrCreateStateSet();
    state->setAttributeAndModes(lw, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
    state->setAttributeAndModes(pm, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
    state->setMode(GL_LIGHTING, osg::StateAttribute::PROTECTED | osg::StateAttribute::OFF);
    state->setMode(GL_LINE_SMOOTH, osg::StateAttribute::OFF);
    state->setMode(GL_BLEND, osg::StateAttribute::ON);

    osg::MatrixTransform* marker_tform = new osg::MatrixTransform;
    marker_tform->setMatrix(pose);
    marker_tform->addChild(marker_geode);

    return marker_tform;
}

osg::Node* MakeTrajectory(const std::vector<cvl::PoseD>& poses,
                          float length,float width,
                          float point_radius,
                          cvl::Vector3d color)
{

    osg::Group* group=new osg::Group();
    for(const auto& pose:poses){
        group->addChild(MakeAxisMarker(pose, length, width));

    }
    if(point_radius<=0) return group;

    std::vector<cvl::Vector3d> xs,cs;
    xs.reserve(poses.size());
    cs.reserve(poses.size());
    for(const auto& pose:poses){
        xs.push_back(pose.getTinW());
        cs.push_back(color);
    }
    group->addChild(MakePointCloud(xs, cs, point_radius));
    // add the points too...
    return group;
}
}
