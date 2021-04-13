#include <assert.h>
#include <iostream>
#include <sstream>


#include <osgViewer/Viewer>
#include <mlib/vis/main_event_handler.h>


using std::cout;using std::endl;
namespace mlib{

MainEventHandler::MainEventHandler(osgViewer::Viewer* viewer)
    : osgGA::GUIEventHandler(), viewer(viewer){
    viewer->setCameraManipulator(cm);
}

bool MainEventHandler::handleKeyEvent(const osgGA::GUIEventAdapter& ea){
    //assert(ea.getEventType()==osgGA::GUIEventAdapter::KEYDOWN);
    //cout<<"key: "<<ea.getKey()<<endl;
    int key=ea.getKey();
    switch(key){

    case 'q':{
        cm->move(0,-step,0);
        break;
    }
    case 'e':{
        cm->move(0,step,0);


        break;}
    case 'w':{
        cm->move(0,0,step);
        break;}
    case 's':{
        cm->move(0,0,-step);

        break;}
    case 'a':{


        cm->move(-step,0,0);

        break;}
    case 'd':{
        cm->move(step,0,0);
        break;}
    case 'o':{
        step*=4.0/3.0;
        break;
    }
    case 'p':{
        step*=0.75;
        break;
    }
    case 49:{//osgGA::GUIEventAdapter::KEY_KP_1:{
        cm->rotate(0,0,-angle_step);
        break;
    }

    case 122:{//z
        cm->rotate(0,0,angle_step);
        break;
    }
    case 51:{//osgGA::GUIEventAdapter::KEY_KP_3:{
        cm->rotate(0,0,angle_step);
        break;
    }
        // c
    case 99:{//c
        cm->rotate(0,0,-angle_step);
        break;
    }
    case 52:{//osgGA::GUIEventAdapter::KEY_KP_4:{
        cm->rotate(0,-angle_step,0);
        break;
    }
    case 54:{//osgGA::GUIEventAdapter::KEY_KP_6:{
        cm->rotate(0,angle_step,0);
        break;
    }
    case 56:{//osgGA::GUIEventAdapter::KEY_KP_6:{
        cm->rotate(angle_step,0,0);
        break;
    }
    case 53:{//osgGA::GUIEventAdapter::KEY_KP_6:{
        cm->rotate(-angle_step,0,0);
        break;
    }
    case ' ':{
        cm->reset();
        step=default_step;
        break;
    }
    case 't':{
        // if in fps mode, hide cursor, and limit to inside, otherwise fix...
        if(in_fps_mode) deactivate_fps_mode();
        else activate_fps_mode();
        break;
    }
    default:
        cout<<"not mapped to action: "<< key<<endl;
        break;
    }

    return true;
}


void MainEventHandler::activate_fps_mode(){
    if(in_fps_mode) return;
    cout<<"activate fps mode"<<endl;

    osgViewer::Viewer::Windows windows;
    viewer->getWindows(windows);
    for(auto &window : windows) {
        window->useCursor(false);
        window->setCursor(osgViewer::GraphicsWindow::NoCursor);
    }
    in_fps_mode=true;
}
void MainEventHandler::deactivate_fps_mode(){
    if(!in_fps_mode) return;
    cout<<"deactivate fps mode"<<endl;
    osgViewer::Viewer::Windows windows;
    viewer->getWindows(windows);
    for(auto &window : windows) {
        window->useCursor(true);
        window->setCursor(osgViewer::GraphicsWindow::InheritCursor);
    }
    in_fps_mode=false;
}

bool MainEventHandler::handle(const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& /*aa*/)
{
    bool redraw = false;
    switch(ea.getEventType()) {
    case(osgGA::GUIEventAdapter::FRAME):  { redraw=(redraw || handleFrameEvent(ea));break; }
    case(osgGA::GUIEventAdapter::KEYDOWN):{ redraw=(redraw || handleKeyEvent(ea));break; }
    case osgGA::GUIEventAdapter::PUSH:{
        if(ea.getButtonMask()!=osgGA::GUIEventAdapter::LEFT_MOUSE_BUTTON) break;
        startx=ea.getXnormalized();
        starty=ea.getYnormalized();

        startp=cm->getPose();
        break;
    }
    case osgGA::GUIEventAdapter::DRAG : {
        if(ea.getButtonMask()!=osgGA::GUIEventAdapter::LEFT_MOUSE_BUTTON) break;
        //cout<<"dragging!"<<endl;
        double x=ea.getXnormalized() - startx;
        double y=ea.getYnormalized() - starty;


        // up in local coordinates!
        cvl::PoseD Pcw=startp;
        cvl::Vector3d up(0,1,0); // in local coordinates
        cvl::Vector3d from(0,0,0); // in local coordinates
        // ideally something that would reproject to where it is
        // as if it was centered
        cvl::Vector3d to(x,y,1);

        // lookat is always well defined!
        cm->set_pose(cvl::lookAt(to,from,up)*Pcw);


        break;
    }
    default:
        //cout<<"got : "<<ea.getEventType()<<endl;
        break;
    }

    if (redraw) {
        // Make changes to the scene graph
        return true;
    }
    return false;
}




bool MainEventHandler::handleFrameEvent([[maybe_unused]]const osgGA::GUIEventAdapter& ea){
    return false;
    // should redraw?
}









}// end namespace mlib
