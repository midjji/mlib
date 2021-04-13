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
# if 1

osg::Matrixd cvl2osg(cvl::Matrix4d m){



    m=Matrix4d(1,0,0,0,
               0,-1,0,0,
               0,0,-1,0,
               0,0,0,1)*m;

        m=m.transpose();

    return osg::Matrixd(m(0, 0), m(0, 1), m(0, 2), m(0, 3),
                        m(1, 0), m(1, 1), m(1, 2), m(1, 3),
                        m(2, 0), m(2, 1), m(2, 2), m(2, 3),
                        m(3, 0), m(3, 1), m(3, 2), m(3, 3));
}
#else
osg::Matrixd cvl2osg(Matrix4d M){
    // this is very very strange!
    auto m=M.transpose(); // assumes p^TM^T ???
    return osg::Matrixd(
                m(0,0), m(0,1), m(0,2), m(0,3),
                m(1,0), m(1,1), m(1,2), m(1,3),
                m(2,0), m(2,1), m(2,2), m(2,3),
                m(3,0), m(3,1), m(3,2), m(3,3));
}
#endif
osg::Matrixd cvl2osg(PoseD m){
    return cvl2osg(m.get4x4());
}


}// end namespace cvl



