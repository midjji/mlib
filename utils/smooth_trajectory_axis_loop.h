#pragma once
#include "mlib/utils/smooth_trajectory.h"
#include "mlib//utils/derivable_trig.h"
namespace cvl {




class AxisSpin:public SmoothTrajectory
{
public:
    Quaternion<double> q;
    double omega;
    AxisSpin(Vector3d axis=Vector3d(1,0,0),
             double omega=0.1):omega(omega){
        axis.normalize();
        q=Vector4d(0,axis[0],axis[1],axis[2]);
    }
    double t0() const override{return 0;};
    double t1() const override{return 1;};
    // interface...
    Vector4d qs(double time, int derivative) const override {
        return std::pow(omega,derivative)*(q.ulog().upow(derivative)*q.upow(time*omega)).q;
    }
    Vector3d ts([[maybe_unused]] double time,[[maybe_unused]]  int derivative) const override{
        return Vector3d(0,0,0);
    }
    bool analytic_derivatives() const override {return true;}
};


class AxisLoop:public SmoothTrajectory
{
public:
    Quaternion<double> q;
    double omega; // omega =1 means
    AxisLoop(Vector3d axis=Vector3d(1,0,0), double omega=0.01):omega(omega){
        axis.normalize();
        q=Vector4d(0,axis[0],axis[1],axis[2]);
    }
    // interface...
    Vector4d qs(double time, int derivative) const override {
        if(derivative==0)
            return Vector4d(1,0,0,0);
        return Vector4d(0,0,0,0);
        return std::pow(omega,derivative)*(
                    q.ulog().upow(derivative)*q.upow(time*omega)).conj().q;
    }
    Vector3d ts(double time,
    int derivative) const override{
        double wa= 2*3.14159265359;
            Vector3d p(dcos(wa,time,derivative),
                       dsin(wa,time,derivative),0);
            return p*10.0;
    }
    double t0() const override{return 0;};
    double t1() const override{return 1;};
    bool analytic_derivatives() const override {return true;}
};


}
