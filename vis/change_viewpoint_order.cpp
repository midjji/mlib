#include <mlib/vis/change_viewpoint_order.h>
#include <mlib/vis/main_event_handler.h>
namespace mlib {
ChangeViewPointOrder::ChangeViewPointOrder(cvl::PoseD Pcw):Order(false),pose(Pcw){}
void ChangeViewPointOrder::event(MainEventHandler* meh){
    meh->cm->set_pose(pose);
}
}
