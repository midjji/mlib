#pragma once
#include <mlib/utils/cvl/pose.h>
#include <osgGA/StandardManipulator>

namespace mlib {


/**
 * @brief The FPSManipulator class
 * first person shooter spectator manipulator,
 * works as you would expect, unlike the osg manipulators, no idea why
 */
class  FPSManipulator : public osgGA::StandardManipulator
{
    cvl::PoseD pose;
public:
    cvl::PoseD getPose() const;
    // camera coordinates
    void set_pose(cvl::PoseD P);

    // in camera coordinates
    void move(double x, double y, double z);
    // in camera coordinates
    void rotate(double ax, double ay, double az);

    void reset();

    FPSManipulator( int flags = DEFAULT_SETTINGS );
    void setByMatrix( const osg::Matrixd& matrix );
    void setByInverseMatrix( const osg::Matrixd& matrix );
    osg::Matrixd getMatrix() const;
    osg::Matrixd getInverseMatrix() const;

    void setTransformation( const osg::Vec3d& eye, const osg::Quat& rotation );
    void setTransformation( const osg::Vec3d& eye, const osg::Vec3d& center, const osg::Vec3d& up );
    void getTransformation( osg::Vec3d& eye, osg::Quat& rotation ) const;
    void getTransformation( osg::Vec3d& eye, osg::Vec3d& center, osg::Vec3d& up ) const;

    void home( const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& us );
    void home( double );

    void init( const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& us );

protected:

    bool handleMouseWheel( const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& us );

    bool performMovementLeftMouseButton( const double eventTimeDelta, const double dx, const double dy );
    bool performMouseDeltaMovement( const float dx, const float dy );
    void applyAnimationStep( const double currentProgress, const double prevProgress );
    bool startAnimationByMousePointerIntersection( const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& us );

};
}
