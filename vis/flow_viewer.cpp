#include <iostream>
#include <thread>
#include <mutex>
#include <atomic>

#include <osgViewer/Viewer>
#include <osgGA/FirstPersonManipulator>
#include <osg/MatrixTransform>

#include "mlib/vis/flow_viewer.h"

#include <mlib/utils/cvl/syncque.h>
#include <mlib/vis/CvGL.h>
#include <mlib/vis/GLTools.h>
#include <mlib/vis/convertosg.h>
using std::cout;
using std::endl;
namespace cvl {

osg::ref_ptr<osg::Geometry> createArrow(Vector3d from,
                                        Vector3d to,
                                        Vector3d color)
{

    //if((from - to).norm()<1e-2)
    osg::ref_ptr<osg::Vec4Array> shared_colors = new osg::Vec4Array;
    shared_colors->push_back(osg::Vec4(color[0],color[1],color[2],0.1f));
    // create Geometry object to store all the vertices and lines primitive.
    osg::ref_ptr<osg::Geometry> polyGeom = new osg::Geometry();

    // note, first coord at top, second at bottom, reverse to that buggy OpenGL image..
    std::vector<Vector3f> xs{
        {0,0,0},
        {1,0,0},
        {0,1,0},
        {1,1,0},
        {0,0,1},
        {1,0,1},
        {0,1,1},
        {1,1,1},
        {1,-0.5f,-0.5f},
        {1, 1.5f,-0.5f},
        {1,-0.5f, 1.5f},
        {1, 1.5f, 1.5f},
        {2, 0.5f, 0.5f},
    };

    // origin point!
    for(auto& x:xs)
        x-=Vector3d(0,0.5,0.5);
    // rescale the damned thing, prettier
    //1,3,5,7+
    double len=4;
    for(auto i:std::vector<int>({1,3,5,7,8,9,10,11,12}))
        xs[i]+=Vector3d(len,0,0);

    // rescale the damned again, according to length of the arrow
    // the arrowhead is normally 1/5 of total... so scale the line ? no
    // scale all of it in length direction! i.e. x!

    // general rescaling
    double scale=(from - to).norm()/(xs[0] - xs[12]).norm();
    for(auto& x : xs)
        x*=scale;
    // check the length!
    //cout<<"arrow: "<<(xs[0] - xs[12]).norm()<<" "<<(from - to).norm()<<endl;

    // rotate so x and z swap...
    double pi=3.14159265359;
    PoseD Pzx=cvl::getRotationMatrixY(-pi/2.0);

    cvl::PoseD P=cvl::lookAt(to, from, Vector3d(0,1,0)).inverse();
    for(auto& x:xs)            x=(P*Pzx*x);

    osg::Vec3 myCoords[] =
    {
        // TRIANGLES 6 vertices, v0..v5
        // note in anticlockwise order.
        osg::Vec3(xs[0][0], xs[0][1],xs[0][2]),
        osg::Vec3(xs[1][0], xs[1][1],xs[1][2]),
        osg::Vec3(xs[2][0], xs[2][1],xs[2][2]),
        osg::Vec3(xs[3][0], xs[3][1],xs[3][2]),
        osg::Vec3(xs[4][0], xs[4][1],xs[4][2]),
        osg::Vec3(xs[5][0], xs[5][1],xs[5][2]),
        osg::Vec3(xs[6][0], xs[6][1],xs[6][2]),
        osg::Vec3(xs[7][0], xs[7][1],xs[7][2]), // 7


        // TRIANGLE STRIP 6 vertices, v6..v11
        // note defined top point first,
        // then anticlockwise for the next two points,
        // then alternating to bottom there after.
        osg::Vec3(xs[8][0], xs[8][1],xs[8][2]),
        osg::Vec3(xs[9][0], xs[9][1],xs[9][2]),
        osg::Vec3(xs[10][0], xs[10][1],xs[10][2]),
        osg::Vec3(xs[11][0], xs[11][1],xs[11][2]),
        osg::Vec3(xs[12][0], xs[12][1],xs[12][2]),


        // TRIANGLE FAN 5 vertices, v12..v16
        // note defined in anticlockwise order.


    };

    int numCoords = sizeof(myCoords)/sizeof(osg::Vec3);

    osg::Vec3Array* vertices = new osg::Vec3Array(numCoords,myCoords);

    // pass the created vertex array to the points geometry object.
    polyGeom->setVertexArray(vertices);

    // use the shared color array.
    polyGeom->setColorArray(shared_colors.get(), osg::Array::BIND_OVERALL);


    // use the shared normal array.
    //polyGeom->setNormalArray(shared_normals.get(), osg::Array::BIND_OVERALL);
    {
        unsigned short myIndices[] =            {                  0,1,2,3,6,7,4,5,0,1            };
        int numIndices = sizeof(myIndices)/sizeof(unsigned short);

        // There are three variants of the DrawElements osg::Primitive, UByteDrawElements which
        // contains unsigned char indices, UShortDrawElements which contains unsigned short indices,
        // and UIntDrawElements which contains ... unsigned int indices.
        // The first parameter to DrawElements is
        osg::ref_ptr<osg::DrawElementsUShort> p0=new osg::DrawElementsUShort(osg::PrimitiveSet::TRIANGLE_STRIP,numIndices,myIndices);
        polyGeom->addPrimitiveSet(p0);
    }
    {
        unsigned short myIndices[] =            {                  8,9,10,11            };
        int numIndices = sizeof(myIndices)/sizeof(unsigned short);
        osg::ref_ptr<osg::DrawElementsUShort> p1=new osg::DrawElementsUShort(osg::PrimitiveSet::TRIANGLE_STRIP,numIndices,myIndices);
        polyGeom->addPrimitiveSet(p1);
    }
    {
        unsigned short myIndices[] =            {                  12,8,9,10,11            };
        int numIndices = sizeof(myIndices)/sizeof(unsigned short);
        osg::ref_ptr<osg::DrawElementsUShort> p2=new osg::DrawElementsUShort(osg::PrimitiveSet::TRIANGLE_FAN,numIndices,myIndices);
        polyGeom->addPrimitiveSet(p2);
    }

    return polyGeom;

}

osg::ref_ptr<osg::Geometry> createArrow(const Flow& flow){
    return createArrow(flow.origin, flow.velocity + flow.origin, flow.color);
}
osg::ref_ptr<osg::Node> MakePointCloud(const PointCloud& points){
    return MakeGenericPointCloud(points.points, points.colors);
}

osg::ref_ptr<osg::Node> MakeTrajectory(const std::vector<PoseD>& poses, float length,float width)
{
    osg::ref_ptr<osg::Group> group=new osg::Group();
    for(auto pose:poses)
        group->addChild(MakeAxisMarker(length,width,
                                       cvl2osg(pose.inverse().get4x4().transpose())));
    return group;
}
osg::ref_ptr<osg::Node> createFlowField(std::shared_ptr<FlowField> ff){
    ff->clean();


    osg::ref_ptr<osg::Geode> field = new osg::Geode();
    // add the flow arrows
    {
        // create the Geode (Geometry Node) to contain all our osg::Geometry objects.
        osg::ref_ptr<osg::Geode> geode = new osg::Geode();

        for(const Flow& flow:ff->flows){

                geode->addDrawable(createArrow(flow));

        }


        // add the points geometry to the geode.

        // Turn off the lighting on the geode.  This is not required for setting up
        // the geometry.  However, by default, lighting is on, and so the normals
        // above are used to light the geometry.
        // - With lighting turned off, the geometry has the same color
        //   regardless of the angle you view it from.
        // - With lighting turned on, the colors darken as the light moves
        //   away from the normal.
        geode->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF | osg::StateAttribute::PROTECTED);
        field->addChild(geode);
    }
    //cout<<"ff->points.colors.size(): "<<ff->points.colors.size()<<endl;
    field->addChild(MakePointCloud(ff->points));
    for(Trajectory tr:ff->trajectories)
        field->addChild(MakeTrajectory(tr.poses,1,1));
    return field;
}



class FPS :public osgGA::FirstPersonManipulator{
public:
    void forward( const double distance );
    void right( const double distance );
    void up( const double distance );
    void reset();
};
void FPS::forward( const double distance ){
    this->moveForward(distance);
}
void FPS::right( const double distance ){
    this->moveRight(distance);
}
void FPS::up( const double distance ){
    this->moveUp(distance);
}
void FPS::reset(){
    _eye.set(0,0,0);
    _rotation.set(0,1,0,0);
}


class MainEventHandler : public osgGA::GUIEventHandler
{

public:

    osgViewer::Viewer* viewer; // not owned by this

    osg::Quat currRot, startRot, endRot;
    osg::Vec3d currPos, startPos, endPos;
    int step;
    int numSteps;

    std::vector<osg::Matrixd> cameraMatrices;

    MainEventHandler(osgViewer::Viewer* viewer)
        : osgGA::GUIEventHandler(), viewer(viewer), step(0), numSteps(0){}




    bool handle(const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& /*aa*/){
        bool redraw = false;

        switch(ea.getEventType()) {
        case(osgGA::GUIEventAdapter::FRAME):  {      redraw=(redraw || handleFrameEvent(ea));break;       }
        case(osgGA::GUIEventAdapter::KEYDOWN):{      redraw=(redraw || handleKeyEvent(ea));break;}
        default:
            break;
        }
        if (redraw) {
            // Make changes to the scene graph
            return true;
        }
        return false;
    }

private:
    double velocity=4;
    std::vector<char> valid= { 'q', 'w', 'e', 'a', 's', 'd', ' ', 'o', 'p' };
    bool hasEffect(char key){
        for(char val:valid){
            if(key==val)
                return true;
        }
        return false;
    }
    bool handleKeyEvent(const osgGA::GUIEventAdapter& ea){
        assert(ea.getEventType()==osgGA::GUIEventAdapter::KEYDOWN);
        if(!hasEffect(ea.getKey()))
            return false;

        FPS* cm = dynamic_cast<FPS*>(viewer->getCameraManipulator());
        assert(cm!=nullptr);
        //cout<<"key: "<<ea.getKey()<<endl;

        switch(ea.getKey()){

        case 'q':{

            cm->up(velocity);
            break;
        }
        case 'e':{
            cm->up(-velocity);
            break;}
        case 'w':{
            cm->forward(velocity);
            break;}
        case 's':{
            cm->forward(-velocity);
            break;}
        case 'a':{
            cm->right(-velocity);
            break;}
        case 'd':{
            cm->right(velocity);
            break;}
        case 'o':{
            velocity*=2.0;
            break;
        }
        case 'p':{
            velocity*=0.5;
            break;
        }
        case ' ':{
            cm->reset();
            velocity=1;
            break;
        }
        default:
            break;
        }
        return true;
    }
    bool handleFrameEvent([[maybe_unused]] const osgGA::GUIEventAdapter& ea){return false;}
};


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
    void update(std::shared_ptr<FlowField> flows){
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
        viewer.setCameraManipulator(new FPS());

        osg::ref_ptr<MainEventHandler> meh=new MainEventHandler(&viewer);
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
        for(auto fv:fvs)
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

