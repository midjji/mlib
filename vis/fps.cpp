#include <mlib/utils/mlog/log.h>
#include <mlib/vis/convertosg.h>
#include <mlib/vis/fps.h>
#include <mlib/utils/cvl/lookat.h>

using namespace osg;
using namespace osgGA;
using std::cout;
using std::endl;

namespace mlib {





// in camera coordinates
void FPSManipulator::move(double x, double y, double z){
    pose=cvl::PoseD(-cvl::Vector3d(x,y,z))*pose;
}
// in camera coordinates
void FPSManipulator::rotate(double ax, double ay, double az){
    pose=cvl::PoseD(cvl::getRotationMatrixXYZ(ax,ay,az))*pose;
    cout<<pose<<endl;
}
// camera coordinates
void FPSManipulator::set_pose(cvl::PoseD P){
    pose=P;
}
void FPSManipulator::set_pose2(cvl::PoseD P){
    pose=cvl::PoseD(cvl::Matrix3d(1,0,0,0,-1,0,0,0,-1))*P;
}
void FPSManipulator::reset(){


    set_pose(cvl::PoseD(cvl::Matrix3d(1,0,0,0,-1,0,0,0,-1))*cvl::PoseD::Identity());
}








/// Constructor.
FPSManipulator::FPSManipulator( int flags )
    : StandardManipulator( flags ){
    setVerticalAxisFixed(false);
}


/** Get the position of the manipulator as 4x4 matrix.*/
Matrixd FPSManipulator::getMatrix() const {

    return cvl2osg(pose.inverse()); // my pose is Pcw, theirs is Pwc
}


/** Get the position of the manipulator as a inverse matrix of the manipulator,
    typically used as a model view matrix.*/
Matrixd FPSManipulator::getInverseMatrix() const {
    // mlog()<<"\n";
    //cout<<pose.get4x4()<<endl;
    return cvl2osg(pose);// my pose is Pcw, theirs is Pwc

}



/** Set the position of the manipulator using a 4x4 matrix.*/
void FPSManipulator::setByMatrix( const Matrixd& matrix )
{
cout<<"set by matrix"<<endl;
    // set variables
    pose.set_t(cvl::osg2cvl(matrix.getTrans()));
    pose.set_q(cvl::osg2cvl(matrix.getRotate()));
    //pose=pose.inverse();
}


/** Set the position of the manipulator using a 4x4 matrix.*/
void FPSManipulator::setByInverseMatrix( const Matrixd& matrix )
{
cout<<"set by inv matrix"<<endl;
    setByMatrix( Matrixd::inverse( matrix ) );
}


// doc in parent
void FPSManipulator::setTransformation(
        [[maybe_unused]] const osg::Vec3d& eye,
[[maybe_unused]]const osg::Quat& rotation )
{
cout<<"set transformation"<<endl;
    pose.set_t(cvl::osg2cvl(eye));
    pose.set_q(cvl::osg2cvl(rotation));
}


// doc in parent
void FPSManipulator::getTransformation( osg::Vec3d& eye, osg::Quat& rotation ) const
{
    cout<<"get transformation"<<endl;
    //mlog()<<"\n";
    eye=cvl::cvl2osg(pose.t());
    rotation=cvl::cvl2osgq(pose.q());
}


// doc in parent
void FPSManipulator::setTransformation( const osg::Vec3d& eye,
                              const osg::Vec3d& center,
                              const osg::Vec3d& up )
{

    pose=cvl::lookAt(cvl::osg2cvl(center),cvl::osg2cvl(eye),cvl::osg2cvl(up));

    //cout<<pose<<endl;
    //pose=PoseD();

}


// doc in parent
void FPSManipulator::getTransformation([[maybe_unused]] osg::Vec3d& eye,
[[maybe_unused]] osg::Vec3d& center,
[[maybe_unused]] osg::Vec3d& up ) const
{

}

void FPSManipulator::home( double currentTime )
{
    StandardManipulator::home( currentTime );
}


void FPSManipulator::home( const GUIEventAdapter& ea, GUIActionAdapter& us )
{
    StandardManipulator::home( ea, us );
}


void FPSManipulator::init( const GUIEventAdapter& ea, GUIActionAdapter& us )
{
    StandardManipulator::init( ea, us );
}




// doc in parent
bool FPSManipulator::handleMouseWheel([[maybe_unused]] const GUIEventAdapter& ea,
[[maybe_unused]] GUIActionAdapter& us )
{


    return true;
}
cvl::PoseD FPSManipulator::getPose() const{return pose;}

// doc in parent
bool FPSManipulator::performMovementLeftMouseButton(
        [[maybe_unused]] const double /*eventTimeDelta*/,
[[maybe_unused]] const double dx,
[[maybe_unused]] const double dy )
{
    /*
    cout<<dx<< " "<<dy<<endl;
    cvl::Vector3d forward=getPose()*cvl::Vector3d(dx,dy,1);
    cvl::Vector3d up(0,1,0);
    pose=cvl::lookAt(forward,cvl::Vector3d(0,0,0),up);
    */
    return true;
}


bool FPSManipulator::performMouseDeltaMovement(
        [[maybe_unused]] const float dx,
[[maybe_unused]] const float dy )
{

    return true;
}



void FPSManipulator::applyAnimationStep(
        [[maybe_unused]] const double currentProgress,
[[maybe_unused]]const double /*prevProgress*/ )
{

}


// doc in parent
bool FPSManipulator::startAnimationByMousePointerIntersection(
        [[maybe_unused]]const osgGA::GUIEventAdapter& ea,
[[maybe_unused]] osgGA::GUIActionAdapter& us )
{

    return true;
}
}
