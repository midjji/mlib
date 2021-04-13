#include <mlib/vis/change_viewpoint_order.h>
#include <mlib/vis/main_event_handler.h>
namespace mlib {
ChangeViewPointOrder::ChangeViewPointOrder():Order(false){}
void ChangeViewPointOrder::event(MainEventHandler* meh){
    meh->cm->set_pose(pose);
}
}
