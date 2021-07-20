#include <mlib/vis/convertosg.h>
namespace cvl{




osg::Vec3d cvl2osg(Vector3d v){
    return osg::Vec3d(v(0),v(1),v(2));
}
osg::Vec3f cvl2osgf(Vector3d v){
    return osg::Vec3f(float(v(0)),float(v(1)),float(v(2)));
}

osg::Quat cvl2osgq(Vector4d q){
    return osg::Quat(q(1),q(2),q(3),q(0));
}

Vector3d osg2cvl(osg::Vec3d v){
    return Vector3d(v[0],v[1],v[2]);
}
Vector4d osg2cvl(osg::Quat q){
    return Vector4d(q[1],q[2],q[3],q[0]);
}


osg::Matrixd cvl2osg(cvl::Matrix4d m){


// flip the y direction and the z direction
    m=cvl::Matrix4d(1,0,0,0,
               0,-1,0,0,
               0,0,-1,0,
               0,0,0,1)*m;
   // return cvl2osg(PoseD(m));

        m=m.transpose(); // why? this inverts the rotation, but also places the translation in a wierd place
    //std::cout<<"convert osg: \n" <<m<<std::endl;
#warning" this is probably wrong in a very weird way"
    return osg::Matrixd(m(0, 0), m(0, 1), m(0, 2), m(0, 3),
                        m(1, 0), m(1, 1), m(1, 2), m(1, 3),
                        m(2, 0), m(2, 1), m(2, 2), m(2, 3),
                        m(3, 0), m(3, 1), m(3, 2), m(3, 3));
}


osg::Matrixd cvl2osg(PoseD p){


//osg::Matrixd M;
  //  M.makeLookAt(cvl2osg(p.rotate(Vector3d(0,0,1))),cvl2osg(p.getTinW()),cvl2osg(p.rotate(Vector3d(0,1,0))));
    //return M;
    //return makeLookAt (const Vec3d &eye, const Vec3d &center, const Vec3d &up)
    return cvl2osg(p.get4x4());
}


}// end namespace cvl



