#include <mlib/vis/flow_field.h>
#include <mlib/vis/axis_marker.h>
#include <mlib/vis/arrow.h>
#include <mlib/vis/point_cloud.h>
#include <osg/Geode>

using namespace cvl;
namespace mlib {


Trajectory::Trajectory(std::vector<PoseD> poses):poses(poses){}

void Trajectory::apply_transform(PoseD pose){
    for(auto& pose_:poses)
        pose=pose*pose_;
}
bool Trajectory::is_normal() const{
    for(PoseD pose:poses)
        if(!pose.is_normal())return false;
    return true;
}



PointCloud::PointCloud(std::vector<Vector3d> points,
                       std::vector<Vector3d> colors):points(points), colors(colors){}

void PointCloud::apply_transform(PoseD pose){
    for(auto& point:points)
        point=pose*point;
}
void PointCloud::clean(){
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
void PointCloud::append(PointCloud pc){
    for(auto p:pc.points)
        points.push_back(p);
    for(auto c:pc.colors)
        colors.push_back(c);
}


FlowField::FlowField(std::vector<Flow> flows,
                     PointCloud points,
                     std::vector<Trajectory> trajectories):flows(flows),points(points),trajectories(trajectories){}

void FlowField::apply_transform(PoseD pose){
    for(Flow& f:flows)
        f.apply_transform(pose);
    points.apply_transform(pose);
    for(Trajectory& t:trajectories)
        t.apply_transform(pose);
}
void FlowField::cap_velocity(double len){
    clean();
    for(auto& flow:flows)
        if(flow.velocity.norm()>len)
            flow.velocity=flow.velocity.normalized()*len;

}
void FlowField::clean(){
    std::vector<Trajectory> trs=trajectories;trajectories.clear();

    for(const auto& tr:trs)
        if(tr.is_normal())
            trajectories.push_back(tr);


    points.clean();
    std::vector<Flow> tmp=flows;
    flows.clear();
    for(Flow flow:tmp)
        if(flow.is_normal())
            flows.push_back(flow);
}
void FlowField::append(std::shared_ptr<FlowField> ff){
    for(Flow f:ff->flows)
        flows.push_back(f);
    points.append(ff->points);
    for(const auto& tr:ff->trajectories)
        trajectories.push_back(tr);
    clean();
}
FlowOrder::FlowOrder(const FlowField& ff, bool update):mlib::Order(update),ff(ff){
    this->ff.clean();
}
osg::Node* FlowOrder::group(){
    this->ff.clean();
    return createFlowField(ff, scale);
}

osg::Node* createFlowField(const FlowField& ff, double marker_scale){



    osg::Geode* field = new osg::Geode();
    // add the flow arrows
    {
        // create the Geode (Geometry Node) to contain all our osg::Geometry objects.
        osg::Geode* geode = new osg::Geode();

        for(const Flow& flow:ff.flows){
            geode->addChild(vis::create_arrow(flow));
            //geode->addDrawable();
        }


        // add the points geometry to the geode.

        // Turn off the lighting on the geode.  This is not required for setting up
        // the geometry.  However, by default, lighting is on, and so the normals
        // above are used to light the geometry.
        // - With lighting turned off, the geometry has the same color
        //   regardless of the angle you view it from.
        // - With lighting turned on, the colors darken as the light moves
        //   away from the normal.
        geode->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF | osg::StateAttribute::PROTECTED);
        field->addChild(geode);
    }

    field->addChild(MakePointCloud(ff.points.points, ff.points.colors,marker_scale));
    for(const Trajectory& tr:ff.trajectories)
        field->addChild(MakeTrajectory(tr.poses,marker_scale,marker_scale));
    return field;
}
}
