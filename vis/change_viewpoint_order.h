#pragma once
#include <mlib/vis/order.h>
#include <mlib/utils/cvl/pose.h>
namespace mlib {
class MainEventHandler;
struct ChangeViewPointOrder:public Order{
    cvl::PoseD pose;
    ChangeViewPointOrder(cvl::PoseD Pcw);
    void event(MainEventHandler* meh) override;

};
}
