#pragma once
#include <vector>
#include <mlib/utils/cvl/pose.h>

#include <mlib/utils/mlog/log.h>
#include <mlib/utils/cvl/quaternion.h>
namespace cvl{



class SmoothTrajectory
{


public:
    // this pose trajectory is smooth and has no fast changes
    virtual ~SmoothTrajectory(){}




    std::vector<double> interior_times(int samples_per_second=10, int border=0) const;
    PoseD operator()(double time) const;
    PoseD pose(double time) const;



    void check_num(){
        // derivative of q must be orthogonal to q,
        // derivative of derivative of q must be ?

    }


    // interface...
    virtual double t0() const=0;
    virtual double t1() const=0;
    double get_first_time() const{return t0();}
    double get_last_time() const{return t1();}
    double first_valid_time() const{return t0();}
    double last_valid_time() const{return t1();}
    // for stuff dependent on control point count!
    int get_first([[maybe_unused]] double time) const{return 0;}
    int get_last([[maybe_unused]] double time) const{return (t1()-t0());} // assume 1 per second.
    int get_last2([[maybe_unused]] double time) const{return (t1()-t0());} // assume 1 per second.

    int current_first_control_point()const{return t0();}
    int current_last_control_point()const{return t1();}
    int size() const{return t1()-t0();}
    double delta_time(){return 1;}

    virtual Vector4d qs(double time, int derivative) const=0;
    virtual Vector3d ts(double time, int derivative) const=0;




    Vector<Quaternion<double>,3> qdot(double time) const
    {
        return Vector<Quaternion<double>,3>(qs(time,0),
                                            qs(time,1),
                                            qs(time,2));
    }
    Vector3d translation(double time, int derivative) const
    {
        return ts(time,derivative);
    }
    Vector<Vector3d,3> translations(double time)const{
        return Vector<Vector3d,3>(ts(time,0),ts(time,1),ts(time,2));
    }


    Quaternion<double> qdot(double time, int derivative) const
    {
        return qdot(time)[derivative];
    }

    template<class T> static Vector3<T>
    angular_velocity(Quaternion<T> q,
                     Quaternion<T> q_dt){
        auto w=(q_dt*q.conj())*T(2.0);
        return w.vec();
    }


    template<class T> static Vector3<T>
    angular_acceleration(Vector<Quaternion<T>,3> qs){
        // body
        auto w=(qs[2]*qs[0].conj() + qs[1]*qs[1].conj())*T(2.0);
        assert(ceres::abs(w(0))<T(1e-12));
        //w(0)=T(0.0); // MUST BE TRUE FOR UNIT Q
        return w.vec();
    }
    virtual Vector3d
    angular_velocity(double time) const{
        auto qs = qdot(time);
        return angular_velocity(qs[0],qs[1]);
    }
    virtual Vector3d
    angular_acceleration(double time) const{
        auto qs = qdot(time);
        return angular_acceleration(qs);
    }

    std::string display() const{
        return "numerically generated smooth trajectory";
    }

protected:
    double delta=1e-3;

};


class XAxis:public SmoothTrajectory{
public:

    // interface...
    double t0() const{return 0;//-100
                     }
    double t1() const{return 100;// 200
                     }

    Quaternion<double> w=Quaternion<double>(Vector4d(0.0,3.1415/2.0,0,0));
    Quaternion<double> q0=Quaternion<double>(Vector4d(0.0,1,0,0));

    double alpha(double time, int derivative) const{

        //int t=time/100;        time -=t*100; // wrap around fix...
        if(derivative==0) return 2.0*time/100.0;
        if(derivative==1) return 2.0/100;
        return 0;
    }

    Vector4d qs(double time, int derivative) const
    {
        if(derivative==0)
        {
            // time in [0,100], during which 1 full rotation occurs.
            double a=alpha(time,0)*3.1415/2.0;


            auto q=q0.upow(alpha(time,0));


            return q.q;
            return Vector4d(std::cos(a),std::sin(a),0,0);
        }
        if(derivative==1){
            auto v=w;
            return ((v*Quaternion<double>(qs(time,0)))*alpha(time,1)).q;
        }
        if(derivative==2){
            double a1=alpha(time,1);
            double a2=alpha(time,2);
            auto v=w;
            return  (v*Quaternion<double>(qs(time,0))*a2 +
                     v*v*Quaternion<double>(qs(time,0))*a1*a1).q;
        }
        return Vector4d(0,0,0,0);
    }


    Vector3d ts([[maybe_unused]]double time, int derivative) const{
        if(derivative==0)
            return Vector3d(0,0,10);
        return Vector3d(0,0,0);
    }
    Vector3d tws(double time, int derivative) const{
        if(derivative==0){
            return PoseD(qs(time,0),ts(time,0)).getTinW();
        }


        return (tws(time+delta,derivative-1) -
                tws(time-delta,derivative-1))/(2.0*delta);
    }

};

class AllRot : public SmoothTrajectory
{
public:
    // interface...
    double t0() const override{return -60;}
    double t1() const override{return 300;}
    constexpr static int degree(){return 4;}

    virtual Vector4d qs(double time, int derivative) const override;
    virtual Vector3d ts(double time, int derivative) const override;


    PoseD pose(double time) const;

    PoseD pose2(double time) const;

};

/**
 * @brief smooth_interpolator
 * @param t
 * @return
 *
 * this is 0 untill about 1, then smoothly increases to 1 at about 2, after which it is 1
 * derivatives are available...
 *
 */
double smooth_interpolator(double t);


class AllRot2 : public SmoothTrajectory
{
public:
    // interface...
    double t0() const override{return -60;}
    double t1() const override{return 300;}
    constexpr static int degree(){return 4; // technically its 5
                                 }
    std::vector<PoseD> display_poses(int per_second=10, int border=0) const;
    std::vector<PoseD> display_poses(std::vector<double> ts) const;


    double s(double t) const; // smooth interpolator!
    Vector3d p(double time) const;


    virtual Vector4d qs(double time, int derivative) const override;
    virtual Vector3d ts(double time, int derivative) const override;


    // gives poses Pwc!
    PoseD pose(double time) const;
    PoseD pose2(double time) const;

};




}
