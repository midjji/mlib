#pragma once

namespace osg{class Node;}

namespace mlib{



struct Order{
    Order(bool update);
    virtual ~Order();
    // returns nullptr if nothing is to be done.
    virtual osg::Node* group(double marker_scale);
    // if true, added to existing, otherwize the scene is cleared first;
    virtual bool is_update();
    bool update=false;

};

}
