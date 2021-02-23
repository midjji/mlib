
#pragma once
#include <mlib/utils/cvl/pose.h>
#include <osgGA/StandardManipulator>
using cvl::PoseD;

/** FPS2 is base class for camera control based on position
    and orientation of camera, like walk, drive, and flight manipulators. */
class  FPS2 : public osgGA::StandardManipulator
{
       PoseD pose;
    public:
       PoseD getPose() const{return pose;}
       // in camera coordinates
       void move(double x, double y, double z);
       // in camera coordinates
       void rotate(double ax, double ay, double az);
       // camera coordinates
       void set_pose(PoseD P);
       void reset();

        FPS2( int flags = DEFAULT_SETTINGS );
        virtual void setByMatrix( const osg::Matrixd& matrix );
        virtual void setByInverseMatrix( const osg::Matrixd& matrix );
        virtual osg::Matrixd getMatrix() const;
        virtual osg::Matrixd getInverseMatrix() const;

        virtual void setTransformation( const osg::Vec3d& eye, const osg::Quat& rotation );
        virtual void setTransformation( const osg::Vec3d& eye, const osg::Vec3d& center, const osg::Vec3d& up );
        virtual void getTransformation( osg::Vec3d& eye, osg::Quat& rotation ) const;
        virtual void getTransformation( osg::Vec3d& eye, osg::Vec3d& center, osg::Vec3d& up ) const;

        virtual void home( const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& us );
        virtual void home( double );

        virtual void init( const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& us );

    protected:

        virtual bool handleMouseWheel( const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& us );

        virtual bool performMovementLeftMouseButton( const double eventTimeDelta, const double dx, const double dy );
        virtual bool performMouseDeltaMovement( const float dx, const float dy );
        virtual void applyAnimationStep( const double currentProgress, const double prevProgress );
        virtual bool startAnimationByMousePointerIntersection( const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& us );

};
