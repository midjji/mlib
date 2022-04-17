#pragma once
#include <memory>
#include <vector>
#include <mlib/utils/cvl/pose.h>
namespace osg{class Node;}

namespace mlib{

class MainEventHandler;
/**
 * @brief The Order struct
 * The point cloud viewer performs one order at a time, and other threads may interrupt
 * so stack up to make them atomic
 */
struct Order{    
    bool clear_scene;
    Order(bool clear_scene=true,int last_n_index=0);
    virtual ~Order();
    // stack up orders

    void push_back(std::unique_ptr<Order> order);
    void process_events(MainEventHandler* meh);

    // returns nullptr if nothing is to be done.
    osg::Node* aggregate_groups();

    // override one or both of:
    // returns nullptr if nothing is to be done.
    virtual osg::Node* group();
    // defaults to nothing
    virtual void event(MainEventHandler* meh);
    int last_n_index(){return last_n_index_;}

private:
    int last_n_index_;
    std::vector<std::unique_ptr<Order>> orders;
};


}
