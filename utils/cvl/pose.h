#pragma once
/* ********************************* FILE ************************************/
/** \file    pose.h
 *
 * \brief    This header contains the Pose<T> class which represents 3D rigid transforms as a unit quaternion and translation
 *
 * \remark
 * - c++11
 * - no dependencies
 * - header only
 * - cuda enabled
 * - tested by test_pose.cpp
 *
 * Initialization using a rotation matrix is allowed but not ideal. Such conversions can give errors if the input isnt a rotation.
 *
 * \todo
 * - how to treat initialization by non rotation matrixes or nans? non rotation matrixes will be convertet to a rotation, but not in a good way. Nans give nans.
 * - convert from implicitly forming the rotation matrix to using quaternion algebra
 *
 *
 *
 *
 * \author   Mikael Persson
 * \date     2014-04-01
 * \note MIT licence
 *
 ******************************************************************************/

#include <mlib/utils/cvl/matrix.h>
#include <mlib/utils/cvl/rotation_helpers.h>
#include <mlib/utils/cvl/quaternion.h>

namespace cvl{

/**
 * @brief The Pose<T> class
 * Quaternion represented rigid transform
 * and simplifies managing rotations by wrapping a quaternion
 *
 *
 * Initialization on bad rotation will assert
 * or give undefined behaviour( often the closest projection
 * onto the manifold of unit quaternions/rotations
 *
 *
 *  In the following, the Quaternions are laid out as 4-vectors, thus:

   q[0]  scalar part.
   q[1]  coefficient of i.
   q[2]  coefficient of j.
   q[3]  coefficient of k.


 q=(cos(alpha/2),sin(alpha/2)N)
 N= rotation axis
 *
 * This is the ceres& mathematical default
 *  but opengl libraries sometimes change the order of quaternion elements
 * There is no intrinsic direction of a transform Xa = Pab*Xb
 * Always specify a,b
 *
 *
 *
 *
 *
 * Tested by posetest.cpp
 */

template<class T>
class Pose {
public:


    T& operator[](uint index){
        assert(index<8);
        if(index<4)
            return q[index];
        return t[index-4];
    }
    mlib_host_device_
    /**
         * @brief Pose initializes as a identity transform, trouble, this prevents std::trivial!
         */

    Pose():q{T(1.0),T(0.0),T(0.0),T(0.0)},t{T(0.0),T(0.0),T(0.0)}{
        // if it wasnt because i rely on the pose() is identity in so many places, this would be a nice fix
        static_assert(std::is_trivially_destructible<Pose<double>>(),"speed");
        static_assert(std::is_trivially_copyable<Pose<double>>(),"speed");
        // the following constraints are good, but not critical
        //static_assert(std::is_trivially_default_constructible<Pose<double>>(),"speed");
        static_assert(std::is_trivially_copy_constructible<Pose<double>>(),"speed");
        //static_assert(std::is_trivially_constructible<Pose<double>>(),"speed");
        static_assert(std::is_trivially_assignable<Pose<double>,Pose<double>>(),"speed");
        //static_assert(std::is_trivial<Pose<double>>(),"speed");

    }
    mlib_host_device_
    static Pose Identity(){return Pose();}

    /**
         * @brief Pose
         * @param Pose<U> converting constructor
         */
    template<class U>
    mlib_host_device_
    Pose(const Pose<U>& p){        q=Vector4<T>(p.q);        t=Vector3<T>(p.t);    }
    Pose(Vector<T,7> v):q(Vector4<T>(v[0],v[1],v[2],v[3])),t(Vector3<T>(v[4],v[5],v[6])){}


    mlib_host_device_
    /**
         * @brief Pose
         * @param q_ unit quaternion
         * @param t_
         */
    Pose(const Vector4<T>& q_, const Vector3<T>& t_):q(q_),t(t_){}

    mlib_host_device_
    /**
         * @brief Pose copies
         * @param q_ unit quaternion pointer
         * @param t_
         */
    explicit Pose(const T* q_, const T* t_, [[maybe_unused]] bool checked){
        for(int i=0;i<4;++i){q[i]=q_[i];} for(int i=0;i<3;++i){t[i]=t_[i];}
    }
    mlib_host_device_
    /**
         * @brief Pose copies
         * @param qt
         */
    explicit Pose(const T* qt, [[maybe_unused]] bool checked){
        for(int i=0;i<4;++i){q[i]=qt[i];} for(int i=0;i<3;++i){t[i]=qt[i+4];}
    }

    // user must verify that the matrix is a rotation separately
    mlib_host_device_
    /**
         * @brief Pose
         * @param R Rotation matrix
         * @param t_
         */
    Pose (const Matrix3<T>& R, const Vector3<T>& t_):q(getRotationQuaternion(R).normalized()),t(t_){

        assert(q.isnormal());
        assert(t.isnormal());
        assert(is_normal());
    }
    mlib_host_device_
    /**
         * @brief Pose translation 0
         * @param R rotation matrix
         */
    Pose (const Matrix3<T>& R):q(getRotationQuaternion(R).normalized()),t{0.0,0.0,0.0}{ }
    // Rotation is i<T>entity
    mlib_host_device_
    /**
         * @brief Pose identity rotation assumed
         * @param t_ translation vector
         */
    Pose (const Vector3<T>& t_):q{T(1.0),T(0.0),T(0.0),T(0.0)},t(t_){}

    mlib_host_device_
    /**
         * @brief Pose
         * @param P a 3x4 matrix with a top left part beeing a rotation matrix
         */
    Pose (const Matrix34<T>& P):q(getRotationQuaternion(P.getRotationPart())),t(P.getTranslationPart()){}
    mlib_host_device_
    /**
         * @brief Pose
         * @param P a 4x4 matrix with a top left part beeing a rotation matrix
         */
    Pose (const Matrix4<T>& P):q(getRotationQuaternion(P.getRotationPart())),t(P.getTranslationPart()){}


    mlib_host_device_
    static Pose<T> eye(){
        return Pose(Vector<T,4>(1.0,0.0,0.0,0.0),Vector<T,3>(0.0,0.0,0.0));
    }

    /**
         * @brief getRRef
         * @return a pointer to the first element of the quaternion in the pose
         */
    mlib_host_device_
    /**
         * @brief getRRef pointer to the quaternion
         * @return
         */
    T* getRRef() {return &q[0];}
    /**
         * @brief getTRefJ
         * @return a pointer to the first element of the translation in the pose
         */
    mlib_host_device_
    /**
         * @brief getTRef pointer to the translation
         * @return
         */
    T* getTRef(){return &t[0];}
    mlib_host_device_
    /**
         * @brief setT set the translation vector, note this is in the camera coordinate system
         * @param t_
         */
    void setT(const Vector3<T>& t_){t=t_;}
    mlib_host_device_
    /**
         * @brief setQuaternion set the quaternion, note t is in the new coordinate system
         * @param q_
         */
    void setQuaternion(const Vector4<T>& q_){q=q_;}




    mlib_host_device_
    /**
         * @brief operator * applies the transform on the point
         * @param ins
         * @return
         *
         * based on the quaternion rotation of ceres, well why though? they basically just create a rotation matrix && apply it...
         */
    Vector3<T> operator*(const Vector3<T>& ins) const{
        return QuaternionRotate(q,ins) + t;
    }

    Vector4<T> operator*(const Vector4<T>& ray){
        // fast variant would be
        //return get4x4()*(ray.dehom()).homogeneous();
        // correct variant would be
        auto v3=QuaternionRotate(q,Vector3<T>(ray[0],ray[1],ray[2])) + ray[3]*t;
        //auto v3=rotate(Vector3<T>(ray[0],ray[1],ray[2])) + ray[3]*t;
        return {v3[0],v3[1],v3[2],ray[3]};
    }

    mlib_host_device_
    /**
         * @brief operator * apply the pose from the left!
         * @param rhs
         * @return
         */
    Pose<T> operator*(const Pose<T>& rhs) const{
        //return Pose(get4x4()*rhs.get4x4());
        return Pose(QuaternionProduct(q,rhs.q),
                    QuaternionRotate(q,rhs.t)+t);
    }
    mlib_host_device_
    /**
         * @brief inverse, note uses that the rotation inverse is its transpose
         * @return
         */
    Pose<T> inverse() const{
        //assert(isnormal());
        //Matrix3<T> Ri=getR().transpose();
        Vector<T,4> qi=conjugateQuaternion(q);

        Vector3<T> ti=-QuaternionRotate(qi,t);
        return Pose(qi,ti);
    }
    mlib_host_device_
    /**
         * @brief invert, note uses that the rotation inverse is its transpose
         * @return
         */
    void invert() {
        Pose<T> p=inverse();
        q=p.q;        t=p.t;
    }
    mlib_host_device_
    /**
         * @brief getR
         * @return the rotation matrix
         */
    Matrix3<T> getR() const{

        return getRotationMatrix(q);
    }
    mlib_host_device_
    /**
         * @brief rotation
         * @return the rotation matrix
         */
    Matrix3<T> rotation() const{return getR();}
    mlib_host_device_
    /**
         * @brief noRotation is the rotation matrix identity
         * @return
         */
    bool noRotation() const{
        if(q[0]!=1) return false;
        for(int i=1;i<4;++i)
            if(q[i]!=0) return false;
        return true;
    }
    mlib_host_device_
    /**
         * @brief isIdentity
         * @return is the pose a identity transform
         */
    bool isIdentity() const{
        if(!noRotation()) return false;
        for(int i=0;i<3;++i)
            if(t[i]!=0) return false;
        return true;
    }
    mlib_host_device_
    /**
         * @brief getT
         * @return the translation
         */
    Vector3<T> getT() const{return t;}
    mlib_host_device_
    /**
         * @brief translation
         * @return the translation
         */
    Vector3<T> translation() const{return t;}
    mlib_host_device_
    /**
         * @brief scaleT applies a scaling to the translation
         * @param scale
         */
    void scaleT(T scale){t*=scale;}
    mlib_host_device_
    /**
         * @brief get3x4
         * @return the 3x4 projection matrix corresp to the rigid transform
         */
    Matrix3<T> get3x4() const{return get3x4(getR(),getT());}
    mlib_host_device_
    /**
         * @brief get4x4
         * @return the 4x4 maxtrix rep of the rigid transform
         */
    Matrix4<T> get4x4() const{return ::cvl::get4x4(getR(),getT());}


    /**
         * @brief getAngle
         * @return the angle of the rotation in radians         
         */
    T getAngle() const
    {
        // assumes unit quaternion!
        const T sin_squared_theta = q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
        const T sin_theta = std::sqrt(sin_squared_theta);
        const T& cos_theta = q[0];
        const T theta =(cos_theta < T(0.0)) ? std::atan2(-sin_theta, -cos_theta)
                                            : std::atan2(sin_theta, cos_theta);        
        return T(2.0)*theta;

        //if(std::abs(q[0]-1)<1e-6) return 0;        return 2.0*std::acos(q[0]);
    }
    T getAngleDegrees(){ // visualization only
        double a=getAngle()*180.0/3.14159265359;
        if(a<0) a+=360.0;
        return a;
    }

    /// get the position of th     //time+=delta_t*0.5;e camera center in world coordinates
    mlib_host_device_
    /**
         * @brief getTinW
         * @return the camera center in world coordinates
         */
    Vector3<T> getTinW() const{        return inverse().t;            }

    mlib_host_device_
    /**
         * @brief essential_matrix
         * @return the normalized essential matrix. This function defines the definition of E used.
         */
    Matrix3<T> essential_matrix() const{
        Vector3<T> t_=getT();
        t_.normalize();
        Matrix3<T> E=t_.crossMatrix()*getR();
        assert(E.isnormal());
        T max=E.absMax();
        E/=max;
        return E;
    }
    mlib_host_device_
    /**
         * @brief distance
         * @param p
         * @return distance between two coordinate system centers
         */
    T distance(const Pose<T>& p) const{
        Pose<T> Pab=(*this)*p.inverse();return Pab.t.length();
    }
    /**
         * @brief angleDistance
         * @param p
         * @return the positive angle between two coordinate systems
         */
    T angleDistance(const Pose<T>& p) const{
        Pose<T> Pab=(*this)*p.inverse();
        double angle=Pab.getAngle();
        if(angle<0)return -angle;
        return angle;
    }

    T geodesic(Pose<T> b){
        return geodesic_vector(b).norm();
    }
    Vector<T,6> geodesic_vector(Pose<T> b) // component wize makes it convenient as residual
    {
        Vector3<T> v=Quaternion<T>(q).geodesic_vector(b.q);
        Vector3<T> p=(b.inverse()*(*this)).t; // could probably be computed faster...
        return Vector<T,6>(v[0],v[1],v[2],
                p[0],p[1],p[2]);

    }
    /// returns true if no value is strange
    mlib_host_device_
    /**
         * @brief isnormal
         * @return true if the pose contains no nans or infs and the quaternion is a unit quaternion
         */
    bool is_normal() const{
        if (!q.isnormal()) return false;
        if (!t.isnormal()) return false;
        if(q.length()-1.0>1e-5) return false;
        return true;
    }


    mlib_host_device_
    /**
         * @brief getQuaternion
         * @return the quaternion
         */
    Vector4<T> getQuaternion() const{return q;}
    mlib_host_device_
    /**
         * @brief normalize ensures that the quaternion length is 1, helpful to counter numerical errors
         */
    void normalize(){
        q.normalize();
    }


    mlib_host_device_
    /**
         * @brief rotateInplace rotates but does not translate the point
         * @param x
         */
    void rotateInplace(Vector3<T>& x) const{
        QuaternionRotate(q,x);
        //auto R=getR();        x=R*x;
    }
    mlib_host_device_
    /**
         * @brief rotate
         * @param x
         * @return  the rotated but not translated vector
         */
    Vector3<T> rotate(const Vector3<T>& x) const{
        return QuaternionRotate(q,x);
    }

    //private:
    /// the unit quaternion representing the rotation, s,i,j,k
    Vector4<T> q;
    /// the translation: x' = (R(q)x) +t
    Vector3<T> t;
    // sizeof Vector3 is 4*sizeof(T)
    //T filler=T(0.0);
    Vector<T,7> getqt() const{
        return Vector<T,7>(q[0],q[1],q[2],q[3],t[0],t[1],t[2]);
    }

    std::string str() const{
        std::stringstream ss;
        ss<<getqt();
        return ss.str();
    }

};

template<class T> Pose<T> lookAt(const Vector3<T>& point,
                                 const Vector3<T>& from,
                                 const Vector3<T>& up0){
    assert((point-from).absSum()>0);
    assert(point.isnormal());
    assert(from.isnormal());
    assert(up0.isnormal());
    // opt axis is point
    Vector3<T> up=up0;            up.normalize();
    Vector3<T> z=point -from;     z.normalize();
    Vector3<T> s=-z.cross(up);    s.normalize();
    Vector3<T> u=z.cross(s);     u.normalize();

    // u=cross f,s
    Matrix3d R(s[0],s[1],s[2],
            u[0],u[1],u[2],
            z[0],z[1],z[2]);
    assert(R.isnormal());
    return Pose<T>(R,-R*from);
}
/// convenience alias for the standard pose
typedef Pose<double> PoseD;

template<class T>
/**
 * @brief operator << a human readable pose description
 * @param os
 * @param t
 * @return
 */
std::ostream& operator<<(std::ostream& os, const Pose<T>& pose){
    return os<<pose.str();
}



}// en<T> namespace cvl


