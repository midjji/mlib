#pragma once
#include <mlib/utils/cvl/pose.h>

namespace cvl{

template<class T>
/**
 * @brief lookAt
 * @param point
 * @param from
 * @param up
 * @return Pose
 *
 * P.inverse()*(from - point)
 *
 */
Pose<T> lookAt(Vector3<T> point,
                                 Vector3<T> from,
                                 Vector3<T> up) {

    // the pose is x_c=Pcw*x_w
    // point up and from are in w
#if 1
    // This lookat is bizzarre,
    // but I think it returns a lookat that works for osg,
    // which probably means its rotated (1,0,0,0,-1,0,0,0,-1)
    assert((point-from).absSum()>0);
    assert(point.isnormal());
    assert(from.isnormal());
    assert(up.isnormal());
    // opt axis is point -from
    up.normalize();
    Vector3<T> z=point -from;     z.normalize();
    Vector3<T> s=-z.cross(up);    s.normalize();
    Vector3<T> u=z.cross(s);     u.normalize();

    // u=cross f,s
    Matrix3d R(s[0],s[1],s[2],
            u[0],u[1],u[2],
            z[0],z[1],z[2]);
    assert(R.isnormal());
    Pose<T> P(R,-R*from);





#else
    // lookat above is utterly bizzare,TODO:  figure out why!

    // opt axis
    Vector3<T> z=point -from;
    if(z.norm()==0) z=Vector3d(0,0,1);
    // (0,0,1) = Rz, osv ... => R^T = (s,up,z)
    z.normalize();
    up=up-up.dot(z)*up; // up orthogonal to z
    if(up.norm()==0) up=Vector3d(z[2],z[1],z[0]).cross(z);
    up.normalize();
    Vector3<T> s=-z.cross(up);
    s.normalize();

    // u=cross f,s
    Matrix3<T> R=Matrix3<T>(s[0],up[0],z[0],
            s[1],up[1],z[1],
            s[2],up[2],z[2]).transpose();




   Pose<T> P(R,-R*from);
   // tests


#endif
//std::cout<<(P.getTinW() - from).norm()<<std::endl;
//std::cout<<"R*z= (0,0,1) = "<<R*z<<std::endl;
return P;
}


}// end namespace cvl



