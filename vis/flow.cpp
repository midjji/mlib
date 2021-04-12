#include <mlib/vis/flow.h>
namespace cvl {
Flow::Flow(Vector3d origin,
     Vector3d velocity,
     Vector3d color):origin(origin),velocity(velocity),color(color){}

void Flow::apply_transform(PoseD pose){
    origin=pose*origin;
    velocity=pose.getR()*velocity;
}
bool Flow::is_normal() const{
    return (origin+velocity+color).isnormal();
}
}
