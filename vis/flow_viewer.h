#pragma once
#include <mlib/utils/cvl/pose.h>
#include <memory>
#include <vector>
#include <osg/Geometry>
namespace cvl {
class Flow{
public:
    Flow(){}
    Flow(Vector3d origin,
         Vector3d velocity,
         Vector3d color):origin(origin),velocity(velocity),color(color){}
    Vector3d origin,velocity,color;
    void apply_transform(PoseD pose){
        origin=pose*origin;
        velocity=pose.getR()*velocity;
    }
    bool is_normal(){
        return (origin+velocity+color).isnormal();
    }
};
class Trajectory{
public:
    Trajectory(){}
    Trajectory(std::vector<PoseD> poses):poses(poses){}
    std::vector<PoseD> poses; // tramsforms from world to camera
    // additional camera information?
    void apply_transform(PoseD pose){
        for(auto& pose_:poses)
            pose=pose*pose_;
    }
    bool is_normal(){
        for(PoseD pose:poses)
            if(!pose.is_normal())return false;
        return true;
    }

};
class PointCloud{
public:
    PointCloud(){}
    ~PointCloud(){}
    PointCloud(std::vector<Vector3d> points,
               std::vector<Vector3d> colors):points(points), colors(colors){}
    std::vector<Vector3d> points, colors;
    void apply_transform(PoseD pose){
        for(auto& point:points)
            point=pose*point;
    }
    void clean(){
        std::vector<Vector3d> p=points;points.clear();
        std::vector<Vector3d> c=colors;colors.clear();
        for(uint i=0;i<p.size();++i){
            if(!p[i].isnormal()) continue;
            if(!c[i].isnormal()) continue;
            if(p[i].norm()>100000) continue;
            points.push_back(p[i]);
            colors.push_back(c[i]);
        }
    }
    void append(PointCloud pc){
        for(auto p:pc.points)
            points.push_back(p);
        for(auto c:pc.colors)
            colors.push_back(c);
    }
};
class FlowField{
public:
    FlowField(){}
    FlowField(std::vector<Flow> flows,
              PointCloud points,
              std::vector<Trajectory> trajectories):flows(flows),points(points),trajectories(trajectories){}
    void apply_transform(PoseD pose){
        for(Flow& f:flows)
           f.apply_transform(pose);
        points.apply_transform(pose);
        for(Trajectory& t:trajectories)
            t.apply_transform(pose);
    }
    void clean(){
        std::vector<Trajectory> trs=trajectories;trajectories.clear();

        for(auto tr:trs)
            if(tr.is_normal())
                trajectories.push_back(tr);


        points.clean();
        std::vector<Flow> tmp=flows;
        flows.clear();
        for(Flow flow:tmp)
            if(flow.is_normal())
                flows.push_back(flow);
    }
    void append(std::shared_ptr<FlowField> ff){
        for(Flow f:ff->flows)
            flows.push_back(f);
        points.append(ff->points);
        for(auto tr:ff->trajectories)
            trajectories.push_back(tr);
    }

    std::vector<Flow> flows;
    PointCloud points;
    std::vector<Trajectory> trajectories;
};


osg::ref_ptr<osg::Geometry> createArrow(const Flow& flow);
/**
 * @brief show_flow
 * @param flows
 * @param name
 * updates the flowviewer with name name or creates a new one with that name.
 * returns immiedately,
 * may take a moment to update
 * thread safe!
 */

void show_flow(std::shared_ptr<FlowField> flows, std::string name="");
void show_flow(std::vector<Flow> flows, std::string name="");

bool flow_field_viewers_open();
void show_trajectory(std::vector<PoseD> ps, std::string name="");
}
