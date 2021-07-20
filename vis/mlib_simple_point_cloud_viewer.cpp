#include <string>
#include <iostream>
#include <thread>
#include <osgViewer/Viewer>
#include <osgViewer/config/SingleWindow>


#include <mlib/utils/cvl/pose.h>
#include "mlib/utils/string_helpers.h"
#include <mlib/utils/mlibtime.h>

#include "mlib/vis/mlib_simple_point_cloud_viewer.h"
#include "mlib/vis/main_event_handler.h"
#include <mlib/vis/change_viewpoint_order.h>
#include <mlib/vis/axis_marker.h>

using std::cout;
using std::endl;
using namespace cvl;
namespace mlib{





PointCloudViewer::PointCloudViewer(std::string name):name(name){}
PointCloudViewer::~PointCloudViewer(){
    cout<<"calling pcv destructor"<<endl;
    running=false;
    queue.notify_all();
    if(thr.joinable()) thr.join();
    cout<<"calling pcv destructor joined"<<endl;
}

void PointCloudViewer::setPointCloud(const std::vector<Vector3d>& xs,
                                     const std::vector<Color>& cols,
                                     double coordinate_axis_length){

    setPointCloud(xs,cols,std::vector<PoseD>(), coordinate_axis_length);
}

void PointCloudViewer::setPointCloud(const std::vector<PoseD>& poses, double coordinate_axis_length){

    std::vector<Vector3d> xs;
    std::vector<Color> cols;
    setPointCloud(xs,cols,poses, coordinate_axis_length);
}

void PointCloudViewer::setPointCloud(
        const std::vector<std::vector<PoseD>>& poses,
        double coordinate_axis_length){

    std::vector<Color> cols;cols.reserve(poses.size());

    for(uint i=0;i<poses.size();++i){
        cols.push_back(Color::nr(i));
    }
    setPointCloud(poses, cols, coordinate_axis_length);
}

void PointCloudViewer::setPointCloud(const std::vector<Vector3d>& xs, const std::vector<PoseD>& poses,
                                     double coordinate_axis_length){
    std::vector<Color> cols;cols.reserve(xs.size());
    for(uint i=0;i<xs.size();++i)
        cols.push_back(Color::nr(i));
    setPointCloud(xs,cols,poses, coordinate_axis_length);
}

void PointCloudViewer::setPointCloud(
        const std::vector<Vector3d>& xs,
        const std::deque<PoseD>& poses,
        double coordinate_axis_length)
{
    std::vector<Color> cols;cols.reserve(xs.size());

    for(uint i=0;i<xs.size();++i)
        cols.push_back(Color::nr(i));

    std::vector<PoseD> pose_vector;pose_vector.reserve(poses.size());
    for(uint i=0;i<poses.size();++i)
        pose_vector.push_back(poses[i]);

    setPointCloud(xs,cols,pose_vector,coordinate_axis_length);
}



void PointCloudViewer::setPointCloud(
        const std::vector<std::vector<PoseD>>& posess,
        const std::vector<Color>& colors,
        double coordinate_axis_length){
    PC pc;
    pc.posess=posess;
    pc.pose_colors=colors;
    pc.coordinate_axis_length=coordinate_axis_length;
    add(pc);
}
void PointCloudViewer::setPointCloud(const std::vector<Vector3d>& xs,
                                     const std::vector<Color>& cols,
                                     const std::vector<PoseD>& poses,
                                     double coordinate_axis_length){
    PC pc;
    pc.xs=xs;
    pc.xs_cols=cols;
    pc.posess.push_back(poses);
    pc.pose_colors.push_back(Color::nr(0));
    pc.coordinate_axis_length=coordinate_axis_length;
    queue.push(std::make_unique<PCOrder>(pc));
}
void PointCloudViewer::sink_(std::unique_ptr<Order>& order){
    if(order!=nullptr) queue.push(std::move(order));
}
void PointCloudViewer::add(const FlowField& ff, bool clear_scene){
    queue.push(std::make_unique<FlowOrder>(ff,clear_scene));
}
void PointCloudViewer::add(PC pc, bool clear_scene){
    pc.fill_colors();
    queue.push(std::make_unique<PCOrder>(pc, clear_scene));
}


void PointCloudViewer::view(PoseD Pcw){
    queue.push(std::make_unique<ChangeViewPointOrder>(Pcw));
}






sPointCloudViewer PointCloudViewer::start(std::string name){
    //cout << endl << endl << endl << endl << "actually inside the start function" << endl << endl << endl << endl;
    sPointCloudViewer pcv=std::make_shared<PointCloudViewer>(name);
    pcv->thr=std::thread([pcv](){ // why is by copy required here, but not in node?
        pcv->running=true;
        pcv->run();
        pcv->running=false;
    });
    return pcv;
}



void PointCloudViewer::run() {
    osg::ref_ptr<osgViewer::Viewer> viewer = new osgViewer::Viewer;
    osg::ref_ptr<mlib::MainEventHandler> meh = new mlib::MainEventHandler(viewer);

    osg::ref_ptr<osg::Group> scene=new osg::Group;

    std::set<osg::Node*> added;
    { // init stuff
        // setup default scene
        {
            // we allways have this to show where origin is, not removed by clear scene
            scene->addChild(MakeAxisMarker(PoseD::Identity(),2,2));
            osg::Node* n=PCOrder(default_scene()).group();
            added.insert(n); // default gets removed on first add
            scene->addChild(n);
        }

        viewer->setSceneData(scene);

        //viewer->setUpViewInWindow(50, 50, 800, 600);
        viewer->apply(new osgViewer::SingleWindow(50,50,800,600,0));

        {
            std::vector<osgViewer::GraphicsWindow*> windows;
            viewer->getWindows(windows);
            if(windows.size()==0){
                mlog()<<"Problem with generating the windows in osg\n";
                return;
            }
            windows.at(0)->setWindowName(name);
        }

        viewer->addEventHandler(meh);
    }



    viewer->realize();
    while(!viewer->done() && running)    {
        viewer->frame();
        mlib::ScopedDelay sd(1e7); // the loop will always take atleast 10ms
        //mlib::sleep_ms(50);



        // perform all orders that have arrived so far
        std::unique_ptr<Order> order;
        while(queue.try_pop(order) && running)
        {
            //if(queue.size()>10) queue.clear();
            if(order->clear_scene){
                for(auto add:added)
                    scene->removeChild(add);
                added.clear();
            }

            osg::Node* group = order->aggregate_groups();
            if(group!=nullptr){
                added.insert(group);
                scene->addChild(group);
            }
            order->process_events(meh);
        }
    }
    viewer->setDone(true);
}
void PointCloudViewer::close(){    running=false;}
bool PointCloudViewer::is_running(){return running;}

void PointCloudViewer::wait_for_done(){
    while(running){
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}






namespace {
std::mutex pcv_mtx;
std::shared_ptr<std::map<std::string, std::shared_ptr<PointCloudViewer> >> pcvs_=nullptr;
std::shared_ptr<std::map<std::string, std::shared_ptr<PointCloudViewer> >> get_pcvs(){
    if(pcvs_==nullptr)
        pcvs_=std::make_shared<std::map<std::string, std::shared_ptr<PointCloudViewer>>>();
    return pcvs_;
}
}

std::shared_ptr<PointCloudViewer> pc_viewer(std::string name){
    if(name=="") name="unnamed window";
    std::unique_lock<std::mutex> ul(pcv_mtx);
    auto pcvs=get_pcvs();
    auto it=pcvs->find(name);
    if(it!=pcvs->end()){
        assert(it->second!=nullptr);
        if(it->second->is_running()) return it->second;
    }
    auto tmp=PointCloudViewer::start(name);
    (*pcvs)[name] = tmp;
    return tmp;
}
void wait_for_viewers(){
    std::unique_lock<std::mutex> ul(pcv_mtx);
    for(auto& pcv:*get_pcvs())
        pcv.second->wait_for_done();
}














}// end namespace mlib

