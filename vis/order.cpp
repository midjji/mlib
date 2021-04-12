#include <mlib/vis/order.h>
#include <osg/Node>

namespace mlib {
Order::Order(bool update):update(update){}
Order::~Order(){}
osg::Node* Order::group([[maybe_unused]] double marker_scale){return nullptr;};
// if true, added to existing, otherwize the scene is cleared first;
bool Order::is_update(){return false;}
}
