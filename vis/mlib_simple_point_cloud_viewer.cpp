#include <string>
#include <iostream>
#include <thread>
#include <iostream>
#include "mlib/vis/mlib_simple_point_cloud_viewer.h"

//#include <osg/Uniform>

//#include <osg/Geometry>
//#include <osg/Notify>
//#include <osg/MatrixTransform>
//#include <osg/Texture2D>
//#include <osg/Billboard>
//#include <osg/LineWidth>

//#include <osgGA/TrackballManipulator>
//#include <osgGA/FlightManipulator>
//#include <osgGA/DriveManipulator>
//#include <osgGA/StateSetManipulator>
//#include <osgDB/Registry>
//#include <osgDB/ReadFile>
//#include <osgViewer/CompositeViewer>
//#include <osgViewer/ViewerEventHandlers>
//#include <osgUtil/Optimizer>
//#include <osgDB/ReadFile>
//#include <osg/Material>
//#include <osgText/Text>
//#include <osgGA/TrackballManipulator>
//#include <osgGA/FlightManipulator>
//#include <osgGA/StateSetManipulator>
//#include <osgGA/FirstPersonManipulator>
//#include <osgViewer/ViewerEventHandlers>
//#include <osgViewer/config/SingleWindow>
//#include <osgFX/Scribe>
//#include <osg/io_utils>

//#include <osg/BlendFunc>
//#include <osg/Depth>
//#include <osg/Projection>
//#include <osg/PolygonOffset>
//#include <osg/Camera>
//#include <osg/FrontFace>
//#include <osg/Node>
//#include <osg/Geode>
#include <osg/MatrixTransform>
#include <osgViewer/Viewer>





#include "mlib/utils/random.h"
#include <mlib/utils/cvl/pose.h>
#include "mlib/utils/string_helpers.h"

#include "mlib/vis/manipulator.h"
#include <mlib/vis/GLTools.h>
#include <mlib/vis/CvGL.h>
#include <mlib/vis/convertosg.h>
#include <mlib/vis/nanipulator.h>




using std::cout;
using std::endl;
using namespace cvl;
namespace mlib{

PointCloudViewer::PointCloudViewer(){}
PointCloudViewer::~PointCloudViewer(){
    cout<<"calling pcv destructor"<<endl;
    close();
    if(thr.joinable()) thr.join();
    cout<<"calling pcv destructor joined"<<endl;
    // delete is private...
    // so create the ref ptrs finally, and then they go out of scope...
    osg::ref_ptr<osg::Group> gr=scene;
    osg::ref_ptr<osgViewer::Viewer> vv=viewer;
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

void PointCloudViewer::setPointCloud(const std::vector<std::vector<PoseD>>& poses,
                                     double coordinate_axis_length){

    std::vector<Color> cols;cols.reserve(poses.size());

    for(uint i=0;i<poses.size();++i){
        cols.push_back(Color::nr(i));
    }
    setPointCloud(poses, cols, coordinate_axis_length);
}

void PointCloudViewer::setPointCloud(const std::vector<Vector3d>& xs, const std::vector<PoseD>& poses,
                                     double coordinate_axis_length){
    std::vector<Color> cols;
    for(uint i=0;i<xs.size();++i)
        cols.push_back(Color::nr(i));
    setPointCloud(xs,cols,poses, coordinate_axis_length);
}

void PointCloudViewer::setPointCloud(
        const std::vector<Vector3d>& xs,
        const std::deque<PoseD>& poses,
        double coordinate_axis_length)
{
    std::vector<Color> cols;
    for(uint i=0;i<xs.size();++i)
        cols.push_back(Color::nr(i));

    std::vector<PoseD> pose_vector;
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
    set_point_cloud(pc);
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
    set_point_cloud(pc);
}
void PointCloudViewer::set_point_cloud(PC pc){
    queue.push(std::shared_ptr<PCGroupable>(new PCGroupable{pc}));
}
osg::Group* PCGroupable::group(double marker_scale)
{
    osg::Group* group=new osg::Group;

    // add a world coordinate sys:
    {
        Matrix3d R(1,0,0,0,1,0,0,0,1);
        osg::ref_ptr<osg::Node> node=MakeAxisMarker(CvlToGl(R),2,2);// argh this kind of ref ptr is insane!
        group->addChild(node);
    }

    {
        auto& posess=pc.posess;
        auto& colors=pc.pose_colors;

        for(uint i=0;i<posess.size();++i){
            auto poses=posess[i];
            auto col=colors[i];



            osg::ref_ptr<osg::Vec3Array> points = new osg::Vec3Array;points->reserve(poses.size());
            osg::ref_ptr<osg::Vec3Array> colors = new osg::Vec3Array;colors->reserve(poses.size());
            for(const PoseD& pose:poses){

                // argh this kind of ref ptr is insane!
                osg::ref_ptr<osg::Node> node=MakeAxisMarker(CvlToGl(pose.inverse()),
                                                            pc.coordinate_axis_length,1);
                group->addChild(node);

                points->push_back(cvl2osg(pose.getTinW()));
                colors->push_back(osg::Vec3(col.getR()/255.0f,col.getG()/255.0f,col.getB()/255.0f));
            }
            // add the points too...
            osg::ref_ptr<osg::Node> node=MakePointCloud(points, colors, marker_scale);
            group->addChild(node);
        }

    }

    {
        auto& xs=pc.xs;
        auto& cols=pc.xs_cols;
        std::vector<Vector3d> xs2;
        std::vector<Vector3d> colors2;
        if(xs.size()>0){
            assert(xs.size()==cols.size());
            osg::ref_ptr<osg::Vec3Array> points = new osg::Vec3Array;points->reserve(xs.size());
            osg::ref_ptr<osg::Vec3Array> colors = new osg::Vec3Array;colors->reserve(cols.size());

            for(uint i=0;i<xs.size();++i){
                if(!std::isnan(xs[i].sum())){
                    Vector3d xr=xs[i];
                    points->push_back(cvl2osg(xr));
                    xs2.push_back(xr);
                    Color col=cols[i];
                    colors->push_back(osg::Vec3(float(col.getR()/255.0),float(col.getG()/255.0),float(col.getB()/255.0)));
                    colors2.push_back(Vector3d(col.getR(),col.getG(),col.getB()));
                }
            }
            osg::ref_ptr<osg::Node> node=MakePointCloud(points, colors, marker_scale);
            group->addChild(node);
        }
    }
    return group;
}


void PointCloudViewer::set_marker_size(double scale){
    marker_scale = scale;
}

void PointCloudViewer::set_pose(PoseD Pcw){
    //TODO verify multithreaded safety
    // TODO dynamic cast check
    std::vector<osgViewer::View*> views;
    viewer->getViews(views,true);
    for(osgViewer::View* view:views)
        ((FPS2*)(view->getCameraManipulator()))->set_pose(Pcw);
}






sPointCloudViewer PointCloudViewer::start(std::string name){
    //cout << endl << endl << endl << endl << "actually inside the start function" << endl << endl << endl << endl;
    sPointCloudViewer pcv=std::make_shared<PointCloudViewer>();
    pcv->init(name);
    pcv->running=true;
    pcv->thr=std::thread(&PointCloudViewer::run,std::ref((*pcv)));
    return pcv;
}

PC default_scene(){
    PC pc;
    pc.xs.reserve(10000);
    pc.xs_cols.reserve(10000);

    // a floor, green
    // the surrounding circle.
    // a sphere, a box and a pyramid

    for(int r=-50;r<50;++r)
        for(int c=-50;c<50;++c)
        {
            pc.xs.push_back(cvl::Vector3d(5*r,0,5*c));
            pc.xs_cols.push_back(mlib::Color(1,1,1));
        }


    for(uint i=0;i<1000;++i){
        pc.xs.push_back(cvl::Vector3d(mlib::randu(-1,1),mlib::randu(-1,1),mlib::randu(-1,1)));
        pc.xs_cols.push_back(mlib::Color(mlib::randu(0,1),mlib::randu(0,1),mlib::randu(0,1)));
    }

    // far away circle of red by angle
    double N=1000;
    double r=10;
    double pi=3.1415;
    for(int i=0;i<N;++i){
        pc.xs.push_back(cvl::Vector3d(r*cos(2*pi*i/N),0, r*sin(2*pi*i/N)));

        pc.xs_cols.push_back(mlib::Color(i/N,0,0));
    }
    return pc;
}


void PointCloudViewer::init(std::string name){


    scene= new osg::Group;
    viewer = new osgViewer::Viewer();
    viewer->setSceneData(scene);
    viewer->setUpViewInWindow(50, 50, 800, 600);
    viewer->setCameraManipulator(new FPS2());
    {
        std::vector<osgViewer::GraphicsWindow*> windows;
        viewer->getWindows(windows);
        windows.at(0)->setWindowName(name);
    }
    set_point_cloud(default_scene());
    viewer->setSceneData(scene);
    osg::ref_ptr<MainEventHandler> meh=new MainEventHandler(viewer);
    viewer->addEventHandler(meh);
}
void PointCloudViewer::run()
{
    std::set<osg::Group*> added;

    viewer->realize();
    while(!viewer->done() && running){
        viewer->frame();

        std::shared_ptr<Groupable> order;
        if(queue.try_pop(order) && running)
        {
            if(queue.size()>10) queue.clear();
            if(order->clear_scene){
                for(auto add:added)
                    scene->removeChild(add);
                added.clear();
            }
            osg::Group* group = order->group(marker_scale);
            added.insert(group);
            scene->addChild(group);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    running=false;
}
void PointCloudViewer::close(){
    running=false;
    viewer->setDone(true);
}
bool PointCloudViewer::is_running(){return running;}

void PointCloudViewer::wait_for_done(){
    while(is_running()){
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

