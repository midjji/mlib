#include "mlib/utils/random.h"

#include <mlib/vis/pc_order.h>

#include <mlib/vis/GLTools.h>
#include <mlib/vis/CvGL.h>
#include <mlib/vis/axis_marker.h>
#include <mlib/vis/point_cloud.h>
#include <mlib/vis/convertosg.h>

using cvl::Vector3d;
using cvl::PoseD;


namespace mlib {


PC default_scene(){
    PC pc;
    pc.xs.reserve(10000);
    pc.xs_cols.reserve(10000);

    // a floor, green
    // the surrounding circle.
    // a sphere, a box and a pyramid

    for(int r=-50;r<50;++r)
        for(int c=-50;c<50;++c)
        {
            pc.xs.push_back(cvl::Vector3d(5*r,0,5*c));
            pc.xs_cols.push_back(mlib::Color(1,1,1));
        }


    for(uint i=0;i<1000;++i){
        pc.xs.push_back(cvl::Vector3d(randu(-1,1),randu(-1,1),randu(-1,1)));
        pc.xs_cols.push_back(mlib::Color(randu(0,1),randu(0,1),randu(0,1)));
    }

    // far away circle of red by angle
    double N=1000;
    double r=10;
    double pi=3.1415;
    for(int i=0;i<N;++i){
        pc.xs.push_back(cvl::Vector3d(r*cos(2*pi*i/N),0, r*sin(2*pi*i/N)));

        pc.xs_cols.push_back(mlib::Color(i/N,0,0));
    }
    return pc;
}

osg::Node* PCOrder::group(double marker_scale)
{
    osg::Group* group=new osg::Group;

    // add a world coordinate sys:
    cvl::Matrix3d R(1,0,0,0,1,0,0,0,1);
    group->addChild(vis::MakeAxisMarker(PoseD::Identity(),2,2));

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

            group->addChild(vis::MakeAxisMarker(pose, pc.coordinate_axis_length, 1));

            xs.push_back(pose.getTinW());
            cs.push_back(col);
        }
        // add the points too...
        group->addChild(vis::MakePointCloud(xs, cs, marker_scale));
    }



    auto& xs=pc.xs;
    auto& cs=pc.xs_cols;
    if(xs.size()>0)
        group->addChild(vis::MakePointCloud(xs, cs, marker_scale));
    return group;
}


}
