
#include <mlib/utils/cvl/pose.h>
#include "mlib/utils/string_helpers.h"

#include <string>
#include <iostream>

#include <osg/Uniform>
#include <osg/Node>
#include <osg/Geometry>
#include <osg/Notify>
#include <osg/MatrixTransform>
#include <osg/Texture2D>
#include <osg/Billboard>
#include <osg/LineWidth>

#include <osgGA/TrackballManipulator>
#include <osgGA/FlightManipulator>
#include <osgGA/DriveManipulator>
#include <osgGA/StateSetManipulator>


#include <osgDB/Registry>
#include <osgDB/ReadFile>
#include <osgViewer/Viewer>
#include <osgViewer/CompositeViewer>
#include <osgViewer/ViewerEventHandlers>

#include <iostream>

#include <osgUtil/Optimizer>
#include <osgDB/ReadFile>

#include <osg/Material>
#include <osg/Geode>
#include <osg/BlendFunc>
#include <osg/Depth>
#include <osg/Projection>
#include <osg/PolygonOffset>
#include <osg/MatrixTransform>
#include <osg/Camera>
#include <osg/FrontFace>

#include <osgText/Text>

#include <osgGA/TrackballManipulator>
#include <osgGA/FlightManipulator>
#include <osgGA/StateSetManipulator>
#include <osgGA/FirstPersonManipulator>
#include <osgViewer/ViewerEventHandlers>

#include <osgViewer/Viewer>
#include <osgViewer/config/SingleWindow>

#include <osgFX/Scribe>

#include <osg/io_utils>
#include <mlib/vis/GLTools.h>
#include <thread>
#include <iostream>

#include "mlib/vis/manipulator.h"

#include <mlib/vis/CvGL.h>
#include <mlib/vis/convertosg.h>


#include "mlib/utils/random.h"

#include <mlib/vis/nanipulator.h>

#include "mlib/vis/mlib_simple_point_cloud_viewer.h"

using std::cout;
using std::endl;
using namespace cvl;
namespace mlib{

PointCloudViewer::PointCloudViewer(){}
PointCloudViewer::~PointCloudViewer(){
    close();
    if(thr.joinable()) thr.join();
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

    void PointCloudViewer::setPointCloud(const std::vector<Vector3d>& xs, const std::deque<PoseD>& poses,
                                         double coordinate_axis_length){
        std::vector<Color> cols;
        for(uint i=0;i<xs.size();++i)
            cols.push_back(Color::nr(i));

        std::vector<PoseD> pose_vector;
        for(uint i=0;i<poses.size();++i)
            pose_vector.push_back(poses[i]);

        setPointCloud(xs,cols,pose_vector,coordinate_axis_length);
    }

void savePointCloud(const std::string& filename, std::vector<Vector3d>& Xs, std::vector<Vector3d>& colors, std::vector<PoseD>& Ps)
{
    uint n;

    std::ofstream file;

    cout << "writing " << filename << endl;
    file.open(filename, std::ios::binary | std::ios::out);

    n = (uint)Xs.size();
    file.write((char*)&n, sizeof(n));

    printf("-----------------pcl_size:%d\n", n);

    for (uint i = 0; i < n; i++) {
        float px = Xs[i](0);
        float py = Xs[i](1);
        float pz = Xs[i](2);
        file.write((char*)&px, sizeof(px));
        file.write((char*)&py, sizeof(py));
        file.write((char*)&pz, sizeof(pz));
        unsigned char r = colors[i](0);
        unsigned char g = colors[i](1);
        unsigned char b = colors[i](2);
        file.write((char*)&r, sizeof(r));
        file.write((char*)&g, sizeof(g));
        file.write((char*)&b, sizeof(b));
    }

    n = (uint)Ps.size();
    file.write((char*)&n, sizeof(n));
    printf("-----------------num_poses:%d\n", n);

    for (uint i = 0; i < n; i++) {
        const double *q = Ps[i].getRRef();
        const double *t = Ps[i].getTRef();

        file.write((char*)q, 4 *sizeof(q[0]));
        file.write((char*)t, 3 * sizeof(t[0]));
    }

    file.close();

}


void PointCloudViewer::setPointCloud(
        const std::vector<std::vector<PoseD>>& posess,
        const std::vector<Color>& colors,
        double coordinate_axis_length){

    std::vector<osg::ref_ptr<osg::Node>> nodes;nodes.reserve(10000);
    assert(colors.size()==posess.size());
    if(posess.size()==0) return;
    if(posess[0].size()==0) return;
    for(uint i=0;i<posess.size();++i){
        auto poses=posess[i];
        auto col=colors[i];



        osg::ref_ptr<osg::Vec3Array> points = new osg::Vec3Array;points->reserve(poses.size());
        osg::ref_ptr<osg::Vec3Array> colors = new osg::Vec3Array;colors->reserve(poses.size());
        for(const PoseD& pose:poses){

            // argh this kind of ref ptr is insane!
            osg::ref_ptr<osg::Node> node=MakeAxisMarker(CvlToGl(pose.inverse()),
                                                        coordinate_axis_length,1);
            nodes.push_back(node);

            points->push_back(cvl2osg(pose.getTinW()));
            colors->push_back(osg::Vec3(col.getR()/255.0f,col.getG()/255.0f,col.getB()/255.0f));
        }
        // add the points too...


        osg::ref_ptr<osg::Node> node=MakePointCloud(points, colors, 25.0);
        nodes.push_back(node);
    }
    // add a world coordinate sys:
    {
        Matrix3d R(1,0,0,0,1,0,0,0,1);
        osg::ref_ptr<osg::Node> node=MakeAxisMarker(CvlToGl(R),2,2);// argh this kind of ref ptr is insane!
        nodes.push_back(node);
    }

    // savePointCloud(fname, xs2, colors2, poses2);

    if(nodes.size()>0)
        que.push(nodes);

}

void PointCloudViewer::setMarkerSize(const float scale){
    this->marker_scale = scale;
}

void PointCloudViewer::set_pose(PoseD Pcw){
    //TODO verify multithreaded safety
    // TODO dynamic cast check
    std::vector<osgViewer::View*> views;
    viewer->getViews(views,true);
    for(osgViewer::View* view:views)
        ((FPS2*)(view->getCameraManipulator()))->set_pose(Pcw);
}



void PointCloudViewer::setPointCloud(const std::vector<Vector3d>& xs,
                                     const std::vector<Color>& cols,
                                     const std::vector<PoseD>& poses,
                                     double coordinate_axis_length){
    std::vector<osg::ref_ptr<osg::Node>> nodes;
    nodes.reserve(1+poses.size());
    std::vector<Vector3d> xs2;
    std::vector<PoseD> poses2;
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
        osg::ref_ptr<osg::Node> node=MakePointCloud(points, colors, 5.0 * marker_scale);
        nodes.push_back(node);
    }
    double len=coordinate_axis_length;
    for(PoseD pose:poses){
        osg::ref_ptr<osg::Node> node=MakeAxisMarker(CvlToGl(pose.inverse()),len, 4*marker_scale);// argh this kind of ref ptr is insane!
        nodes.push_back(node);
    }
    // add a world coordinate sys:
    {
        osg::ref_ptr<osg::Node> node=MakeAxisMarker(CvlToGl(PoseD::Identity()),8,8*marker_scale);// argh this kind of ref ptr is insane!
        nodes.push_back(node);
    }

    // savePointCloud(fname, xs2, colors2, poses2);

    if(nodes.size()>0)
        que.push(nodes);
}


sPointCloudViewer PointCloudViewer::start(std::string name){
    //cout << endl << endl << endl << endl << "actually inside the start function" << endl << endl << endl << endl;
    sPointCloudViewer pcv=std::make_shared<PointCloudViewer>();
    pcv->init(name);
    pcv->running=true;
    pcv->thr=std::thread(&PointCloudViewer::run,std::ref((*pcv)));
    return pcv;
}
osg::ref_ptr<osg::Node> default_scene(){
    osg::ref_ptr<osg::Vec3Array> points = new osg::Vec3Array;
    osg::ref_ptr<osg::Vec3Array> colors = new osg::Vec3Array;






    // a floor, green
    // the surrounding circle.
    // a sphere, a box and a pyramid

    for(int r=-50;r<50;++r)
        for(int c=-50;c<50;++c)
    {
        points->push_back(osg::Vec3(5*r,0,5*c));
        colors->push_back(osg::Vec3(1,1,1));
    }







    points->reserve(12000);
    colors->reserve(12000);

    for(uint i=0;i<1000;++i){
        points->push_back(osg::Vec3(mlib::randu<double>(-1,1),mlib::randu<double>(-1,1),mlib::randu<double>(-1,1)));
        colors->push_back(osg::Vec3(mlib::randu<double>(0,1),mlib::randu<double>(0,1),mlib::randu<double>(0,1)));
    }

    // far away circle of red by angle
    double N=1000;
    double r=10;
    double pi=3.1415;
    for(double i=0;i<N;++i){
        points->push_back(osg::Vec3d(r*cos(2*pi*i/N),0, r*sin(2*pi*i/N)));

        colors->push_back(osg::Vec3d(i/N,0,0));
    }










    osg::ref_ptr<osg::Node> node=MakePointCloud(points, colors, 5.0);
    return node;
}


void PointCloudViewer::init(std::string name){


    viewer = new osgViewer::Viewer();
    viewer->setSceneData(scene.get());
    viewer->setUpViewInWindow(50, 50, 800, 600);
    viewer->setCameraManipulator(new FPS2());
    {
        std::vector<osgViewer::GraphicsWindow*> windows;
        viewer->getWindows(windows);
        windows.at(0)->setWindowName(name);
    }
    ns.push_back(default_scene());
    scene->addChild(ns.at(0));viewer->setSceneData(scene.get());


    osg::ref_ptr<MainEventHandler> meh=new MainEventHandler(viewer);
    viewer->addEventHandler(meh);
}
void PointCloudViewer::run(){


    viewer->realize();




    while(!viewer->done() && running){


        viewer->frame();
        std::vector<osg::ref_ptr<osg::Node>> tmp;
        if(que.try_pop(tmp) && running)
        {
            if(que.size()>10) que.clear();


            if(tmp.size()==0) continue;
            bool success;
            for(const osg::ref_ptr<osg::Node>& n:ns){
                success= scene->removeChild(n);
                assert(success);
                if(!success) exit(1);
            }
            ns=tmp;

            for(const osg::ref_ptr<osg::Node>& n:ns){
                success= scene->addChild(n);
                assert(success);
                if(!success) exit(1);
            }

        }
        else{
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    que.stop();
}
void PointCloudViewer::close(){
    running=false;
    viewer->setDone(true);
    que.stop();
}
bool PointCloudViewer::is_running(){return running;}

bool PointCloudViewer::isdone(){

    if(viewer==nullptr)
        return true;
    if(viewer->done())
        return true;
    return false;
}
void PointCloudViewer::wait_for_done(){
    while(true &&!isdone()){
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

