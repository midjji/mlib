#include <osg/Group>
#include "mlib/utils/random.h"

#include <mlib/vis/pc_order.h>

#include <mlib/vis/axis_marker.h>
#include <mlib/vis/point_cloud.h>
#include <mlib/vis/convertosg.h>
#include <mlib/utils/mlog/log.h>
#include <mlib/vis/axis_marker.h>



using cvl::Vector3d;
using cvl::PoseD;


namespace mlib {
void PC::fill_colors(){
    if(xs_cols.size()==0) xs_cols.push_back(Color::cyan());
    if(xs_cols.size()!=xs.size()) xs_cols.resize(xs.size(),xs_cols[0]);


    for(int i=pose_colors.size();i<int(posess.size());++i)
    pose_colors.push_back(Color::nr(i));


}
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
PCOrder::PCOrder(const PC& pc,
                 bool clear_scene, int last_n_index):Order(clear_scene, last_n_index),pc(pc){}
osg::Node* PCOrder::group()
{
    osg::Group* group=new osg::Group;

    auto& posess=pc.posess;
    auto& colors=pc.pose_colors;
    int size=std::min(posess.size(), colors.size());
    posess.resize(size);
    colors.resize(size);

    for(uint i=0;i<posess.size();++i) {
        auto& poses=posess[i];
        auto col=colors[i];

        if(poses.size()<1) continue;

        group->addChild(MakeTrajectory(poses,pc.coordinate_axis_length, 1,1,col.cvl()));
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
    if(xs.size()==0) return nullptr;
    osg::Group* group=new osg::Group;

    group->addChild(MakePointCloud(xs, cs, radius));
    return group;
}

}
