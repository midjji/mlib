#pragma once
#include <vector>
#include <mlib/utils/cvl/pose.h>
#include <mlib/utils/mlog/log.h>
#include <mlib/utils/cvl/quaternion.h>
namespace cvl{



class SmoothTrajectory
{
protected:
public:
    // this pose trajectory is smooth and has no fast changes
    virtual ~SmoothTrajectory(){}
    constexpr static int degree(){return 4; }



    std::vector<double> interior_times(int samples_per_second=10, int border=0) const;
    PoseD operator()(double time) const;

    // Should give poses Pwc
    PoseD pose(double time) const;

    // interface...
    virtual double t0() const=0;
    virtual double t1() const=0;
    double get_first_time() const{return t0();}
    double get_last_time() const{return t1();}
    double first_valid_time() const{return t0();}
    double last_valid_time() const{return t1();}
    // for stuff dependent on control point count!
    //int get_first([[maybe_unused]] double time) const{return 0;}
    //int get_last([[maybe_unused]] double time) const{return (t1()-t0());} // assume 1 per second.
    //int get_last2([[maybe_unused]] double time) const{return (t1()-t0());} // assume 1 per second.

    int current_first_control_point()const{return t0();}
    int current_last_control_point()const{return t1();}
    int size() const{return t1()-t0();}
    double delta_time(){return 1;}

    virtual Vector4d qs(double time, int derivative) const =0;
    virtual Vector3d ts(double time, int derivative) const =0;

    virtual bool analytic_derivatives() const{return false;}

    std::vector<PoseD> display_poses(std::vector<double> ts) const;




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
        return qs(time,derivative);
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

std::vector<std::shared_ptr<SmoothTrajectory>> test_trajectories();




#if 0

class AllRot : public SmoothTrajectory
{
public:
    // interface...
    Vector4d qs(double time, int derivative) const override {
        return (q.ulog().upow(derivative)*q.upow(time)).q;
    }
    Vector3d ts([[maybe_unused]] double time,[[maybe_unused]]  int derivative) const override{
        return Vector3d(0,0,0);
    }
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
    double s(double t) const; // smooth interpolator!
    Vector3d p(double time) const;
    // gives poses Pwc!
   PoseD pose2(double time) const override;

};


class AllRot3 : public SmoothTrajectory
{
public:
    // interface...

    Vector3d position(double time) const;

    // gives poses Pwc!

    PoseD pose2(double time) const override;

};

#endif


}
