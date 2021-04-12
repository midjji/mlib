#pragma once
/* ********************************* FILE ************************************/
/** \file    mlib_simple_point_cloud_viewer.h
 *
 * \brief    This header contains the PointCloudViewer class which allows for simple convenient asynch viewing of pointclouds and trajectories with gui
 * Dependencies:
 * - c++11
 * - openscenegraph
 * - opencv
 *
 *
    Use: auto pcv = pc_viewer(std::string name);
    then pcv->set...
    Thats it.
 *
 *
 * \author   Mikael Persson
 * \date     2014-09-01
 * \note MIT licence
 *
 ******************************************************************************/

#include <memory>
#include <vector>
#include "mlib/utils/cvl/syncque.h"
#include "mlib/utils/cvl/pose.h"
#include <mlib/utils/colormap.h>
#include <mlib/stream/sink.h>
#include <mlib/vis/order.h>
#include <mlib/vis/pc_order.h>
#include <mlib/vis/flow_field.h>



namespace osg{
class Group;
}
namespace osgViewer {
class Viewer;
}

namespace mlib{

class PointCloudViewer;
typedef std::shared_ptr<PointCloudViewer> sPointCloudViewer;
class MainEventHandler;





/**
 * @brief The PointCloudViewer class
 *  Does what pcl never could, just show a point cloud given!
 *
 *  Contains a simple gui and convenient management of scale stuff...
 *
 * Create a new one with start,
 * que a new point cloud for display with setPointCloud
 *
 */
class PointCloudViewer :public cvl::Sink<std::shared_ptr<Order>>{
public:
        static sPointCloudViewer start(std::string name="Point Cloud Viewer(wasdqe,mouse)");

    PointCloudViewer(std::string name);
    ~PointCloudViewer();

    void setPointCloud(const std::vector<cvl::Vector3d>& xs,
                       const std::vector<Color>& cols,
                       double coordinate_axis_length=1);

    void setPointCloud(const std::vector<cvl::Vector3d>& xs,
                       const std::vector<Color>& cols,
                       const std::vector<cvl::PoseD>& poses,
                       double coordinate_axis_length=1);

    void setPointCloud(const std::vector<cvl::Vector3d>& xs,
                       const std::vector<cvl::PoseD>& poses,
                       double coordinate_axis_length=1);

    void setPointCloud(const std::vector<cvl::Vector3d>& xs,
                       const std::deque<cvl::PoseD>& poses,
                       double coordinate_axis_length=1);

    void setPointCloud(const std::vector<cvl::PoseD>& poses,
                       double coordinate_axis_length=1);

    void setPointCloud(const std::vector<std::vector<cvl::PoseD>>& poses,
                       double coordinate_axis_length=1);

    void setPointCloud(const std::vector<std::vector<cvl::PoseD>>& poses,
                       const std::vector<Color>& colors,
                       double coordinate_axis_length=1);

    void set(vis::FlowField& ff);
    void set_point_cloud(PC pc);



    void set_marker_size(double scale);
    void set_pose(cvl::PoseD Pcw);


    void wait_for_done();
    void close();
    bool is_running();
private:
    void sink_(std::shared_ptr<Order>& pc) override;

    // scales the points
    std::atomic<double> marker_scale{5.0f};

    void run();
    std::atomic<bool> running;

    std::thread thr;




    cvl::SyncQue<std::shared_ptr<Order>> queue;
    osgViewer::Viewer* viewer=nullptr;
    osg::Group* scene = nullptr;
    mlib::MainEventHandler* meh =nullptr;

};


std::shared_ptr<PointCloudViewer> pc_viewer(std::string name);
// waits untill all named viewers made using the above are done
void wait_for_viewers();







}// end namespace mlib
