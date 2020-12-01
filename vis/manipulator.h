#pragma once
#include <osgViewer/Viewer>
#include <osgGA/FirstPersonManipulator>
#include <mlib/utils/cvl/pose.h>
namespace mlib{
class FPS :public osgGA::FirstPersonManipulator{
public:
    void move(double x, double y, double z);
	void reset();
    void show_state();
    void set_pose(cvl::PoseD Pcw);
};



class MainEventHandler : public osgGA::GUIEventHandler
{

public:

    osg::ref_ptr<osgViewer::Viewer> viewer;
	std::vector<osg::Matrixd> cameraMatrices;

        MainEventHandler(osg::ref_ptr<osgViewer::Viewer> viewer);
	bool handle(const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& /*aa*/);
    bool in_fps_mode=false;

private:
    double default_step=0.1;
    double step=0.1;
    double angle_step=5*3.1415/180.0;
void deactivate_fps_mode();
void activate_fps_mode();

	bool handleKeyEvent(const osgGA::GUIEventAdapter& ea);
        bool handleFrameEvent(const osgGA::GUIEventAdapter& ea);

        double startx,starty;
        cvl::PoseD startp;
};
}// end namespace mlib
