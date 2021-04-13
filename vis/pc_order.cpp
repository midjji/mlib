#include <osg/Group>
#include "mlib/utils/random.h"

#include <mlib/vis/pc_order.h>

#include <mlib/vis/axis_marker.h>
#include <mlib/vis/point_cloud.h>
#include <mlib/vis/convertosg.h>
#include <mlib/utils/mlog/log.h>



using cvl::Vector3d;
using cvl::PoseD;


namespace mlib {


PC default_scene(){
    PC pc;
    pc.xs.reserve(12000);
    pc.xs_cols.reserve(12000);

    // a floor, green
    // the surrounding circle.
    // a sphere, a box and a pyramid

    for(int r=-50;r<50;++r)
        for(int c=-50;c<50;++c)
        {
            pc.xs.push_back(cvl::Vector3d(5*r,0,5*c));
            pc.xs_cols.push_back(mlib::Color(255,255,255));
        }


    for(uint i=0;i<1000;++i){
        pc.xs.push_back(cvl::Vector3d(randu(-1,1),randu(-1,1),randu(-1,1)));
        pc.xs_cols.push_back(mlib::Color(randu(0,255),randu(0,255),randu(0,255)));
    }

    // far away circle of red by angle
    double N=1000;
    double r=10;
    double pi=3.1415;
    for(int i=0;i<N;++i){
        pc.xs.push_back(cvl::Vector3d(r*cos(2*pi*i/N),0, r*sin(2*pi*i/N)));

        pc.xs_cols.push_back(mlib::Color(255*i/N,0,0));
    }
    return pc;
}

osg::Node* PCOrder::group()
{
    osg::Group* group=new osg::Group;

    auto& posess=pc.posess;
    auto& colors=pc.pose_colors;

    for(uint i=0;i<posess.size();++i){

        auto& poses=posess[i];
        auto& col=colors[i];
        if(poses.size()<1) continue;



        std::vector<cvl::Vector3d> xs;xs.reserve(poses.size());
        std::vector<mlib::Color> cs;cs.reserve(poses.size());
        for(const cvl::PoseD& pose:poses){

            // argh this kind of ref ptr is insane!

            group->addChild(MakeAxisMarker(pose, pc.coordinate_axis_length, 1));

            xs.push_back(pose.getTinW());
            cs.push_back(col);
        }
        // add the points too...
        group->addChild(MakePointCloud(xs, cs, scale));
    }
    auto& xs=pc.xs;
    auto& cs=pc.xs_cols;
    if(xs.size()>0)
        group->addChild(MakePointCloud(xs, cs, scale));
    return group;

}

PointsOrder::PointsOrder(const std::vector<cvl::Vector3d>& xs,
                         Color color,
                         bool clear_scene,
                         double radius):Order(clear_scene),xs(xs),radius(radius){
    cs.resize(xs.size(), cvl::Vector3d(color[0],color[1],color[2]));
}

PointsOrder::PointsOrder(const std::vector<cvl::Vector3d>& xs,
                         const std::vector<cvl::Vector3d>& colors, // rgb 0-255
                         bool clear_scene,
                         double radius):Order(clear_scene),xs(xs),cs(colors),radius(radius){
    cvl::Vector3d w=cvl::Vector3d(255,255,255);
    cs.resize(xs.size(), w);
}
PointsOrder::PointsOrder(const std::vector<cvl::Vector3d>& xs,
                         const std::vector<Color>& colors,
                         bool clear_scene,
                         double radius):Order(clear_scene),xs(xs),radius(radius){
    cs.reserve(xs.size());
    for(auto c:colors)cs.push_back(cvl::Vector3d(c[0],c[1],c[2]));
    cs.resize(xs.size(), cvl::Vector3d(255,0,0));
}
osg::Node* PointsOrder::group() {
    osg::Group* group=new osg::Group;

    // add a world coordinate sys:
    group->addChild(MakeAxisMarker(PoseD::Identity(),2,2));

    if(xs.size()>0)
        group->addChild(MakePointCloud(xs, cs, radius));
    return group;
}

}
