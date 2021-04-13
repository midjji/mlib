#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Point>
#include <osgGA/StateSetManipulator>


#include <mlib/vis/convertosg.h>
#include <mlib/vis/point_cloud.h>

namespace mlib {
osg::Node* MakePointCloud(
        const std::vector<cvl::Vector3d>& xs,
        const std::vector<mlib::Color>& cs,
        float radius){
    std::vector<cvl::Vector3d> cols;cols.reserve(cs.size());
    for(auto c:cs)cols.push_back(cvl::Vector3d(c[0],c[1],c[2]));
return MakePointCloud(xs,cols,radius);
}
osg::Node* MakePointCloud(
        const std::vector<cvl::Vector3d>& xs,
        const std::vector<cvl::Vector3d>& cs,
        float radius)
{
    auto cols=cs;
    cols.resize(xs.size(),cvl::Vector3d(0,128,128));

    osg::ref_ptr<osg::Vec3Array> vertices = new osg::Vec3Array;
    osg::ref_ptr<osg::Vec3Array> colors   = new osg::Vec3Array;
    {
        vertices->reserve(xs.size());
        colors->reserve(xs.size());


        cvl::Matrix3d R(1,0,0,
                 0,-1,0,
                 0,0,-1);
        for(auto x:xs) vertices->push_back(cvl2osgf(R*x));

        for(auto c:cols) {
            colors->push_back(cvl2osgf(c/255.0));
        }
    }

    osg::ref_ptr<osg::Geometry> geo = new osg::Geometry;
    geo->setVertexArray(vertices);
    osg::ref_ptr<osg::Vec3Array> normals = new osg::Vec3Array;
    normals->push_back(osg::Vec3(0, 0, 0)); // Dummy value (necessary when using BIND_OFF?)
    geo->setNormalArray(normals, osg::Array::BIND_OFF);
    geo->setColorArray(colors, osg::Array::BIND_PER_VERTEX);
    geo->addPrimitiveSet(new osg::DrawArrays(GL_POINTS, 0, vertices->size()));

    osg::ref_ptr<osg::Geode> geode = new osg::Geode;
    geode->addDrawable(geo);

    osg::PolygonMode *pm = new osg::PolygonMode(osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::POINT);
    osg::Point *psz = new osg::Point(10*radius);

    osg::StateSet *state = geode->getOrCreateStateSet();
    state->setAttributeAndModes(psz, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
    state->setAttributeAndModes(pm, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
    state->setMode(GL_LIGHTING, osg::StateAttribute::PROTECTED | osg::StateAttribute::OFF);
    //state->setMode(GL_POINT_SMOOTH, osg::StateAttribute::ON);
    //state->setMode(GL_BLEND, osg::StateAttribute::ON);


    osg::Group* group=new osg::Group();
    group->addChild(geode);
    return group;
}
}
