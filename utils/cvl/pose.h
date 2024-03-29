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
#include <vector>
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
 * Always specify a,b, its very tempting to make them two ints as template!
 * but compile times would become atrocious...
 * for now use transformdirection struct below
 *
 *
 *
 *
 *
 * Tested by posetest.cpp
 */
template<class T>
class Pose {
    /// the unit quaternion representing the rotation, s,i,j,k
    /// the translation: x' = (R(q)x) +t
    Vector<T,7> data; // q,t
    // padding or not?

public:


    T& operator[](uint index){ return data[index];}
    __host__ __device__
    /**
         * @brief Pose initializes as a identity transform, trouble, this prevents std::trivial!
         */
    Pose():data{T(1.0),T(0.0),T(0.0),T(0.0),T(0.0),T(0.0),T(0.0)}{}
    __host__ __device__
    static Pose Identity(){return Pose();}


    template<class U>

    Pose(const Pose<U>& p):data(p.qt()){}

    __host__ __device__
    Pose(Vector<T,7> v):data(v){}

    __host__ __device__
    Pose(Vector4<T> q, Vector3<T> t):data(q[0],q[1],q[2],q[3],t[0],t[1],t[2]){}


    __host__ __device__
    /**
         * @brief Pose copies
         * @param q_ unit quaternion pointer
         * @param t_
         */
    explicit Pose(const T* q, const T* t, [[maybe_unused]] bool checked){
        for(int i=0;i<4;++i){data[i]=q[i];}
        for(int i=0;i<3;++i){data[4+i]=t[i];}
    }
    __host__ __device__
    /**
         * @brief Pose copies
         * @param data
         */
    explicit Pose(const T* ptr, [[maybe_unused]] bool checked){
        for(int i=0;i<7;++i){data[i]=ptr[i];}
    }

    // user must verify that the matrix is a rotation separately
    __host__ __device__
    /**
         * @brief Pose
         * @param R Rotation matrix
         * @param t_
         */
    Pose(Matrix3<T> R, Vector3<T> t=Vector3<T>::Zero())
    {
        auto q=getRotationQuaternion(R).normalized();
        for(int i=0;i<4;++i){data[i]=q[i];}
        for(int i=0;i<3;++i){data[4+i]=t[i];}
    }

    // Rotation is i<T>entity
    __host__ __device__
    /**
         * @brief Pose identity rotation assumed
         * @param t_ translation vector
         */
    Pose (const Vector3<T>& t):data{T(1.0),T(0.0),T(0.0),T(0.0),t[0],t[1],t[2]}{}

    __host__ __device__
    /**
         * @brief Pose
         * @param P a 3x4 matrix with a top left part beeing a rotation matrix
         */
    Pose (const Matrix34<T>& P){
        auto q=getRotationQuaternion(P.getRotationPart()).normalized();
        auto t=P.getTranslationPart();
        for(int i=0;i<4;++i){data[i]=q[i];}
        for(int i=0;i<3;++i){data[4+i]=t[i];}
    }
    __host__ __device__
    /**
         * @brief Pose
         * @param P a 4x4 matrix with a top left part beeing a rotation matrix
         */
    Pose (const Matrix4<T>& P)
    {
        auto q=getRotationQuaternion(P.getRotationPart()).normalized();
        Vector3<T> t=Vector4<T>(P(0,3),P(1,3),P(2,3),P(3,3)).dehom();

        for(int i=0;i<4;++i){data[i]=q[i];}
        for(int i=0;i<3;++i){data[4+i]=t[i];}
    }


    __host__ __device__
    static Pose<T> eye(){
        return Pose(Vector<T,4>(1.0,0.0,0.0,0.0),Vector<T,3>(0.0,0.0,0.0));
    }


    __host__ __device__
    T* getRRef() {return data.begin();}
    __host__ __device__
    T* getTRef(){return t_begin();}
    T* t_begin(){return &data[4];}
    T* begin(){return data.begin();}
    T* end(){return data.end();}

    __host__ __device__
    void setT(Vector3<T> t){for(int i=0;i<3;++i) data[4+i]=t[i];}
    __host__ __device__
    void setQuaternion(Vector4<T> q){for(int i=0;i<4;++i) data[i]=q[i];}
    __host__ __device__
    void set_t(Vector3<T> t){for(int i=0;i<3;++i) data[4+i]=t[i];}
    __host__ __device__
    void set_q(Vector4<T> q){for(int i=0;i<4;++i) data[i]=q[i];}




    __host__ __device__
    /**
         * @brief operator * applies the transform on the point
         * @param ins
         * @return
         *
         * based on the quaternion rotation of ceres, well why though? they basically just create a rotation matrix && apply it...
         */
    Vector3<T> operator*(const Vector3<T>& ins) const{
        return QuaternionRotate(q(),ins) + t();
    }

    Vector4<T> operator*(const Vector4<T>& ray) const{
        // fast variant would be
        //return get4x4()*(ray.dehom()).homogeneous();
        // correct variant would be
        auto v3=QuaternionRotate(q(),Vector3<T>(ray[0],ray[1],ray[2])) + ray[3]*t();
        return Vector4<T>(v3[0],v3[1],v3[2],ray[3]);
    }

    __host__ __device__
    /**
         * @brief operator * apply the pose from the left!
         * @param rhs
         * @return
         */
    inline Pose<T> operator*(const Pose<T>& rhs) const{
        return Pose(QuaternionProduct(q(),rhs.q()),
                    QuaternionRotate(q(),rhs.t())+t());
    }
    __host__ __device__
    /**
         * @brief inverse,
         * @return
         */
    inline Pose<T> inverse() const{   
        Vector<T,4> qi=conjugateQuaternion(q());

        Vector3<T> ti=-QuaternionRotate(qi,t());
        return Pose(qi,ti);
    }
    inline Vector3<T> apply_inverse(Vector3<T> x){
        require(false,"implementation");
        x-=t();
        quaternionRotate(conjugateQuaternion(q()));
    }

    __host__ __device__
    /**
         * @brief invert, note uses that the rotation inverse is its transpose
         * @return
         */
    void invert() {
        data=inverse().data;
    }
    __host__ __device__
    /**
         * @brief getR
         * @return the rotation matrix
         */
    Matrix3<T> getR() const{
        return getRotationMatrix(q());
    }
    __host__ __device__
    /**
         * @brief noRotation is the rotation matrix identity
         * @return
         */
    bool is_rotation() const{
        if(data[0]!=1) return false;
        for(int i=1;i<4;++i)
            if(data[i]!=0) return false;
        return true;
    }
    __host__ __device__
    /**
         * @brief isIdentity
         * @return is the pose a identity transform
         */
    bool is_identity() const{
        if(data[0]!=1.0) return false;
        for(int i=0;i<7;++i) if(data[i]!=0.0) return false;
        return true;
    }
    __host__ __device__
    /**
         * @brief getT
         * @return the translation
         */
    Vector3<T> getT() const{return t();}
    __host__ __device__
    /**
         * @brief translation
         * @return the translation
         */
    Vector3<T> translation() const{return t();}
    __host__ __device__
    /**
         * @brief scaleT applies a scaling to the translation
         * @param scale
         */
    void scaleT(T scale){
        for(int i=4;i<7;++i)
            data[i]*=scale;
    }
    __host__ __device__
    /**
         * @brief get3x4
         * @return the 3x4 projection matrix corresp to the rigid transform
         */
    Matrix<T,3,4> get3x4() const{
        Matrix<T,3,3> R=getR();
        return ::cvl::get3x4(R,t());
    }
    __host__ __device__
    /**
         * @brief get4x4
         * @return the 4x4 maxtrix rep of the rigid transform
         */
    Matrix4<T> get4x4() const{return ::cvl::get4x4(getR(),getT());}


    /**
         * @brief angle
         * @return the angle of the rotation in radians
         * double the theta for quaternions
         * can be negative!
         */
    T angle() const
    {
        return T(2.0)*unit_quaternion::theta(q());
    }
    T angle_degrees(){ // visualization only
        return angle()*T(180.0/3.14159265359);
    }


    /// get the position of th     //time+=delta_t*0.5;e camera center in world coordinates
    __host__ __device__
    /**
         * @brief getTinW
         * @return the camera center in world coordinates
         */
    Vector3<T> getTinW() const{        return inverse().t();            }

    __host__ __device__
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

    // never create distance(Pose function, it , or geodesic(Pose...
    // it practically guarantees the user will make errors, that wont be compiler caught...


    __host__ __device__
    inline Vector<T,6> geodesic_vector() const
    {
        // implies a scaling of R vs t!
        // never ever use these in optimization! severe numerics issues
        Vector4<T> v=unit_quaternion::log(q());
        return Vector<T,6>(v[1],v[2],v[3],
                data[4],data[5],data[6]);
    }

    /// returns true if no value is strange
    __host__ __device__
    /**
         * @brief isnormal
         * @return true if the pose contains no nans or infs and the quaternion is a unit quaternion
         */
    bool is_normal() const{
        if (!data.isnormal()) return false;
        if(q().length()-1.0>1e-5) return false;
        return true;
    }

    __host__ __device__
    /**
         * @brief normalize ensures that the quaternion length is 1, helpful to counter numerical errors
         */
    void normalize(){
        auto tmp=q();
        tmp.normalize();
        if(tmp[0]<0) tmp*=T(-1);
        for(int i=0;i<4;++i)
            data[i] =tmp[i];
    }
    __host__ __device__
    /**
         * @brief rotate
         * @param x
         * @return  the rotated but not translated vector
         */
    inline Vector3<T> rotate(const Vector3<T>& x) const{
        return QuaternionRotate(q(),x);
    }

    std::string str() const{
        std::stringstream ss;
        ss<<data;
        return ss.str();
    }
    inline Vector4<T> q() const {return Vector4<T>(data[0],data[1],data[2],data[3]);}
    inline Vector3<T> t() const {return Vector3<T>(data[4],data[5],data[6]);}
    inline T& tx()            { return data[4];}
    inline T& ty()            { return data[5];}
    inline T& tz()            { return data[6];}
    inline const T& tx() const{ return data[4];}
    inline const T& ty() const{ return data[5];}
    inline const T& tz() const{ return data[6];}

    inline T& real(){ return data[0];} // sometimes called w
    inline T& qw()  { return data[0];}
    inline T& qx()  { return data[1];}
    inline T& qy()  { return data[2];}
    inline T& qz()  { return data[3];}

    inline const T& real() const{ return data[0];} // sometimes called w
    inline const T& qw()   const{ return data[0];} // sometimes called w
    inline const T& qx()   const{ return data[1];}
    inline const T& qy()   const{ return data[2];}
    inline const T& qz()   const{ return data[3];}


    inline Vector<T,7> qt() const{return data;}


};




template<class T> Pose<T>
///
/// \brief interpolate
/// \param P0
/// \param P1
/// \param time
///
/// interpolate(0)==P0,
/// interpolate(1)==P1
///
interpolate(Pose<T> P0 /*from*/, Pose<T> P1/*to*/, double fraction)
{
    Quaternion<T> q0(P0.q());
    Quaternion<T> q1(P1.q());
    Quaternion<T> q=q0*(q0.conj()*q1).upow(fraction);
    Vector3d prev_t= P0.t();
    Vector3d next_t= P1.t();
    Vector3d t=(next_t - prev_t)*fraction + prev_t;
    return Pose<T>(q.q,t);
}

template<class T>
/**
 * @brief sensible_pose_error_vector
 * @param Paw
 * @param Paw_obs
 * @return
 *
 *
 * The fantastic thing about this metric is that the direction of the poses dont matter!
 */
inline Vector<T,9> sensible_pose_error_vector(const Pose<T>& Paw, const Pose<T>& Paw_obs)
{
    // could create the 3x4 matrix, but this approach is possibly numerically safer
    Vector3<T> e0=Paw*Vector3<T>(T(1),T(0),T(0)) - Paw_obs*Vector3<T>(T(1),T(0),T(0));
    Vector3<T> e1=Paw*Vector3<T>(T(0),T(1),T(0)) - Paw_obs*Vector3<T>(T(0),T(1),T(0));
    Vector3<T> e2=Paw*Vector3<T>(T(0),T(0),T(1)) - Paw_obs*Vector3<T>(T(0),T(0),T(1));
    Vector<T,9> residuals;
    residuals[0]=e0[0];
    residuals[1]=e0[1];
    residuals[2]=e0[2];

    residuals[3]=e1[0];
    residuals[4]=e1[1];
    residuals[5]=e1[2];

    residuals[6]=e2[0];
    residuals[7]=e2[1];
    residuals[8]=e2[2];
    return residuals;
}

template<class T> inline T sensible_pose_error(const Pose<T>& Paw, const Pose<T>& Paw_obs)
{
    return sensible_pose_error_vector(Paw, Paw_obs).norm()/9.0;
}

/// convenience alias for the standard pose
typedef Pose<double> PoseD;


struct TransformDirection{
    struct geodesic_operator
    {
        const bool common_from;
        inline Vector6d operator()(PoseD A, PoseD B) const{
            if (common_from){
                // Pab, Pcb case
                return (A*B.inverse()).geodesic_vector();

            }
            // Pab, Pac case
            return (A.inverse()*B).geodesic_vector();
        }
    };

    // x_a = P_{ab}(x_b)
    uint32_t a=10; // estimates of a are 11-19 and so on...
    uint32_t b=0;
    auto geodesic_vector(const TransformDirection& dir) const{
        if(a==dir.a)
            return geodesic_operator{false};
        if(b!=dir.b) {
            mlog()<<"either A or B must match or you cant combine them! \n";
            exit(1);
        }
        return geodesic_operator{true};
    }
};

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
template<class T>
void scale_translations(std::vector<Pose<T>>& ps, double scale){
    for(auto& p:ps)
        p.scaleT(scale);
}
template<class T>
std::vector<Pose<T>> scaled_translations(const std::vector<Pose<T>>& ps, double scale){
    std::vector<Pose<T>> ret=ps;
    scale_translations(ret,scale);
    return ret;
}
template<class T> std::vector<Pose<T>> invert(const std::vector<Pose<T>>& ps){
    std::vector<Pose<T>> rets;rets.reserve(ps.size());
    for(const PoseD& p:ps)rets.push_back(p.inverse());
    return rets;
}


// if it wasnt because i rely on the pose() is identity in so many places, this would be a nice fix
static_assert(std::is_trivially_destructible<Pose<double>>(),"speed");
static_assert(std::is_trivially_copyable<Pose<double>>(),"speed");
// the following constraints are good, but not critical
//static_assert(std::is_trivially_default_constructible<Pose<double>>(),"speed");
static_assert(std::is_trivially_copy_constructible<Pose<double>>(),"speed");
//static_assert(std::is_trivially_constructible<Pose<double>>(),"speed");
static_assert(std::is_trivially_assignable<Pose<double>,Pose<double>>(),"speed");
//static_assert(std::is_trivial<Pose<double>>(),"speed");





}// en<T> namespace cvl


