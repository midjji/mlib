
#include <mlib/vis/nanipulator.h>
#include <mlib/vis/convertosg.h>
#include <mlib/utils/mlog/log.h>
 

using namespace osg;
using namespace osgGA;
using std::cout;
using std::endl;



// in camera coordinates
void FPS2::move(double x, double y, double z){
    pose=cvl::PoseD(-cvl::Vector3d(x,y,z))*pose;
}
// in camera coordinates
void FPS2::rotate(double ax, double ay, double az){
    pose=pose*cvl::PoseD(cvl::getRotationMatrixXYZ(ax,ay,az));
    cout<<pose<<endl;
}
// camera coordinates
void FPS2::set_pose(cvl::PoseD P){
    pose=P;
}
void FPS2::reset(){
    set_pose(PoseD());
}








/// Constructor.
FPS2::FPS2( int flags )
    : StandardManipulator( flags ){}


/** Get the position of the manipulator as 4x4 matrix.*/
Matrixd FPS2::getMatrix() const {

    return cvl2osg(pose.get4x4().inverse()); // my pose is Pcw, theirs is Pwc
}


/** Get the position of the manipulator as a inverse matrix of the manipulator,
    typically used as a model view matrix.*/
Matrixd FPS2::getInverseMatrix() const {
    // mlog()<<"\n";
    //cout<<pose.get4x4()<<endl;
    return cvl2osg(pose.get4x4());// my pose is Pcw, theirs is Pwc

}



/** Set the position of the manipulator using a 4x4 matrix.*/
void FPS2::setByMatrix( const Matrixd& matrix )
{

    // set variables
    pose.t=cvl::osg2cvl(matrix.getTrans());
    pose.q=cvl::osg2cvl(matrix.getRotate());
}


/** Set the position of the manipulator using a 4x4 matrix.*/
void FPS2::setByInverseMatrix( const Matrixd& matrix )
{

    setByMatrix( Matrixd::inverse( matrix ) );
}


// doc in parent
void FPS2::setTransformation(
        [[maybe_unused]] const osg::Vec3d& eye,
[[maybe_unused]]const osg::Quat& rotation )
{

    pose.t=cvl::osg2cvl(eye);
    pose.q=cvl::osg2cvl(rotation);
}


// doc in parent
void FPS2::getTransformation( osg::Vec3d& eye, osg::Quat& rotation ) const
{
    //mlog()<<"\n";
    eye=cvl::cvl2osg(pose.t);
    rotation=cvl::cvl2osgq(pose.q);
}


// doc in parent
void FPS2::setTransformation( const osg::Vec3d& eye, const osg::Vec3d& center, const osg::Vec3d& up )
{

    pose=cvl::lookAt(cvl::osg2cvl(center),cvl::osg2cvl(eye),cvl::osg2cvl(up));

    //cout<<pose<<endl;
    pose=PoseD();

}


// doc in parent
void FPS2::getTransformation([[maybe_unused]] osg::Vec3d& eye,[[maybe_unused]] osg::Vec3d& center, osg::Vec3d& up ) const
{

}

void FPS2::home( double currentTime )
{
    StandardManipulator::home( currentTime );
}


void FPS2::home( const GUIEventAdapter& ea, GUIActionAdapter& us )
{
    StandardManipulator::home( ea, us );
}


void FPS2::init( const GUIEventAdapter& ea, GUIActionAdapter& us )
{
    StandardManipulator::init( ea, us );
}




// doc in parent
bool FPS2::handleMouseWheel([[maybe_unused]] const GUIEventAdapter& ea,[[maybe_unused]] GUIActionAdapter& us )
{


    return true;
}


// doc in parent
bool FPS2::performMovementLeftMouseButton([[maybe_unused]] const double /*eventTimeDelta*/,[[maybe_unused]] const double dx, const double dy )
{

    return true;
}


bool FPS2::performMouseDeltaMovement([[maybe_unused]] const float dx,[[maybe_unused]] const float dy )
{

    return true;
}



void FPS2::applyAnimationStep([[maybe_unused]] const double currentProgress, [[maybe_unused]]const double /*prevProgress*/ )
{

}


// doc in parent
bool FPS2::startAnimationByMousePointerIntersection(
        [[maybe_unused]]const osgGA::GUIEventAdapter& ea,[[maybe_unused]] osgGA::GUIActionAdapter& us )
{

    return true;
}
