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
#include <osgViewer/Viewer>


namespace mlib{

class PointCloudViewer;
typedef std::shared_ptr<PointCloudViewer> sPointCloudViewer;


class PC{
public:
    std::vector<cvl::Vector3d> xs;
    std::vector<Color>& cols;
    std::vector<cvl::PoseD> ps;
};


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
class PointCloudViewer{
public:

    int frame_count;

    PointCloudViewer();
    ~PointCloudViewer();

    void setPointCloud(const std::vector<cvl::Vector3d>& xs,
                       const std::vector<Color>& cols);
    void setPointCloud(const std::vector<cvl::Vector3d>& xs,
                       const std::vector<Color>& cols,
                       const std::vector<cvl::PoseD>& poses);

    void setPointCloud(const std::vector<cvl::Vector3d>& xs,
                       const std::vector<cvl::PoseD>& poses);

    void setPointCloud(const std::vector<cvl::PoseD>& poses);
    void setPointCloud(const std::vector<std::vector<cvl::PoseD>>& poses);

    void setPointCloud(const std::vector<std::vector<cvl::PoseD>>& poses,
                       const std::vector<Color>& colors);

    void setMarkerSize(const float scale);

    void set_pose(cvl::PoseD Pcw);

    static sPointCloudViewer start(std::string name="Point Cloud Viewer(wasdqe,mouse)");
    bool isdone();
    void wait_for_done();
    void close();
    bool is_running();
private:

    float marker_scale = 1.0;

    void init(std::string name);
    void run();
    std::atomic<bool> running;

    std::thread thr;


    std::vector<osg::ref_ptr<osg::Node>> ns;
    cvl::SyncQue<std::vector<osg::ref_ptr<osg::Node>>> que;



    osg::ref_ptr<osgViewer::Viewer> viewer=nullptr;
    osg::ref_ptr<osg::Group> scene = new osg::Group;

};


std::shared_ptr<PointCloudViewer> pc_viewer(std::string name);
// waits untill all named viewers made using the above are done
void wait_for_viewers();







}// end namespace mlib
