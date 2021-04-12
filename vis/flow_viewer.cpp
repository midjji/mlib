#if 0
#include <iostream>
#include <thread>
#include <mutex>
#include <atomic>

#include <osgViewer/Viewer>
#include <osg/MatrixTransform>

#include "mlib/vis/flow_viewer.h"

#include <mlib/utils/cvl/syncque.h>
#include <mlib/vis/CvGL.h>
#include <mlib/vis/GLTools.h>
#include <mlib/vis/arrow.h>
#include <mlib/vis/convertosg.h>
#include <mlib/vis/nanipulator.h>
#include <mlib/vis/manipulator.h>
using std::cout;
using std::endl;
namespace cvl {







namespace flow_viewer {

class FlowViewer {
public:
    static std::shared_ptr<FlowViewer> create(std::string name){
        auto fv=std::make_shared<FlowViewer>("Flow Viewer: "+name);
        fv->start();
        return fv;
    }
    FlowViewer(std::string name):name(name){}
    ~FlowViewer(){
        stop();
        if(thr.joinable())
            thr.join();
    }
    void stop(){
        running=false;
    }
    bool get_running(){return running;}
    void update(std::shared_ptr<vis::FlowField> flows){
        que.push(flows);
    }
private:
    std::string name;
    osgViewer::Viewer viewer;
    osg::ref_ptr<osg::Group> scene=new osg::Group; // group with one node in it, the current_flow
    osg::ref_ptr<osg::Node> current_flow=nullptr;
    osg::ref_ptr<osg::MatrixTransform> coordinatesystem=nullptr;

    std::thread thr;
    std::atomic<bool> running;
    SyncQue<std::shared_ptr<FlowField>> que;


    void start(){
        running=true;
        thr=std::thread([this] {
            run();
        });
    }

    void update(osg::ref_ptr<osg::Node> flow){
        //cout<<"updating flow"<<endl;
        if(current_flow!=nullptr)
            scene->removeChild(current_flow);
        scene->addChild(flow);
        current_flow=flow;
    }

    void init(){

        viewer.setUpViewInWindow(10,10,1600,900,0);
        viewer.setCameraManipulator(new FPS2());

        osg::ref_ptr<mlib::MainEventHandler> meh=new mlib::MainEventHandler(&viewer);
        viewer.addEventHandler(meh);
        // add a world coordinate sys:
        {
            cvl::Matrix3d R(1,0,0,0,1,0,0,0,1);
            coordinatesystem=MakeAxisMarker(CvlToGl(R),2,2);// argh this kind of ref ptr is insane!
            scene->addChild(coordinatesystem);
        }

        // add model to viewer.
        viewer.setSceneData( scene );
        //viewer.getCamera()->setClearColor(osg::Vec4f(1.0f,1.0f,1.0f,1.0f));
        viewer.realize();

        osgViewer::Viewer::Windows ViewerWindow;
        viewer.getWindows(ViewerWindow);
        if (!ViewerWindow.empty())
            ViewerWindow.front()->setWindowName( name );
    }

    void run()
    {
        init();
        while(running){

            viewer.frame();
            std::shared_ptr<FlowField> tmp;
            if(que.try_pop(tmp))
                update( createFlowField(tmp) );
            else
                std::this_thread::sleep_for(std::chrono::milliseconds(30));
            if(viewer.done())
                running=false;
        }
        viewer.setDone(true);
    }
};


class FlowViewerManager{
public:
    std::shared_ptr<FlowViewer> get_flow_viewer(std::string name){
        std::unique_lock<std::mutex> ul(mtx); // locks

        auto it=fvs.find(name);
        if(it==fvs.end()){
            auto fv=FlowViewer::create(name);
            fvs[name]=fv;
            return fv;
        }
        if(!it->second->get_running()){
            auto fv=FlowViewer::create(name);
            fvs[name]=fv;
            return fv;
        }
        return it->second;
    }
    int get_running_viewer_count(){
        std::unique_lock<std::mutex> ul(mtx); // locks
        int c=0;
        for(const auto &fv:fvs)
            if(fv.second->get_running()) c++;
        return c;
    }
private:
    std::map<std::string, std::shared_ptr<FlowViewer>> fvs;
    std::mutex mtx;
};

std::mutex fvm_mutex;
std::shared_ptr<FlowViewerManager> fvm=nullptr;
std::shared_ptr<FlowViewerManager> get_flow_viewer_manager(){
    std::unique_lock<std::mutex> ul(fvm_mutex); // locks
    if(fvm == nullptr)
        fvm = std::make_shared<FlowViewerManager>();
    return fvm;
}

}

using namespace flow_viewer;



bool flow_field_viewers_open(){
    return get_flow_viewer_manager()->get_running_viewer_count()>0;
}

void show_flow(std::vector<Flow> flows, std::string name){
    auto ff=std::make_shared<FlowField>();
    ff->flows=flows;
    show_flow(ff,name);
}
void show_flow(std::shared_ptr<FlowField> ff, std::string name){
    auto fvm=get_flow_viewer_manager();
    auto fv=fvm->get_flow_viewer(name);
    fv->update(ff);
}
void show_trajectory(std::vector<PoseD> ps, std::string name){
    std::shared_ptr<FlowField> ff=std::make_shared<FlowField>();
    ff->trajectories.push_back(Trajectory(ps));
    show_flow(ff,name);
}
}

#endif
