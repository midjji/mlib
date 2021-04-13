#include <osg/Node>
#include <mlib/vis/order.h>
#include <mlib/vis/main_event_handler.h>

namespace mlib {
Order::Order(bool clear_scene):clear_scene(clear_scene){
    orders.reserve(64);
}
Order::~Order(){}
void Order::process_events(MainEventHandler* meh){
    for(auto& order:orders)
        order->event(meh);
}
void Order::push_back(std::unique_ptr<Order> order){orders.push_back(std::move(order));}
osg::Node* Order::aggregate_groups(){
    osg::Group* n=nullptr;
    for(auto& order:orders){
        osg::Node* g=order->group();
        if(g){
            if(!n) n=new osg::Group;
            n->addChild(g);
        }
    }
        osg::Node* g=group();
        if(g){
             if(!n) n=new osg::Group;
            n->addChild(g);
        }

    return n;
}
osg::Node* Order::group(){return nullptr;};
void Order::event([[maybe_unused]] MainEventHandler* meh){}


}
