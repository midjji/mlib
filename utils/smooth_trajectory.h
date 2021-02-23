#pragma once
#include <vector>
#include <mlib/utils/cvl/pose.h>
#include "mlib/utils/constants.h"
#include <mlib/utils/mlog/log.h>
#include <mlib/utils/cvl/quaternion.h>
namespace cvl{



class SmoothTrajectory
{


public:
    // this pose trajectory is smooth and has no fast changes
    virtual ~SmoothTrajectory(){}



    std::vector<PoseD>  display_poses(int N=1000, int border=0) const;
    std::vector<double> interior_times(int N=1000, int border=0) const;
    PoseD operator()(double time) const{
        return PoseD(qs(time, 0), ts(time,0));
    }


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

    virtual Vector4d qs(double time, int derivative) const=0;
    virtual Vector3d ts(double time, int derivative) const=0;
    virtual Vector3d tws(double time, int derivative) const=0;



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
    Vector<Vector3d,3> translation_world(double time)const{
        return Vector<Vector3d,3>(tws(time,0),tws(time,1),tws(time,2));
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
    double delta=1e-5;

};


class XAxis:public SmoothTrajectory{
public:

    // interface...
    double t0() const{return 0;//-100
               }
    double t1() const{return 100;// 200
               }

    Quaternion<double> w=Quaternion<double>(Vector4d(0.0,pi_d/2.0,0,0));
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
            double a=alpha(time,0)*pi_d/2.0;


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
    double t0() const{return -60;}
    double t1() const{return 300;}
    constexpr static int degree(){return 4;}


    std::vector<PoseD> display_poses(int N=1000, int border=0) const{
        return display_poses(this->interior_times(N,border));
    }

    std::vector<PoseD> display_poses(std::vector<double> ts) const{
        std::vector<PoseD> ps;ps.reserve(ts.size());
        // we assume its in x_world=P(t)x_camera, so invert for display?
        for(auto t:ts)
            ps.push_back(pose(t).inverse());
        return ps;
    }


    virtual Vector4d qs(double time, int derivative) const{
        if(derivative==0) return pose(time).q;

        auto a=qs(time+delta,derivative-1);
        auto b=qs(time-delta,derivative-1);
        Vector4d qd= a-b;
        if((a+b).norm()<(a-b).norm())
            qd=a+b;
        qd=qd/(2.0*delta);
        if(derivative==1){
            Vector4d q = pose(time).q;
            // make it orthogonal to q
            qd=qd - qd.dot(q)*q;
            if(std::abs(qd.dot(q))>1e-6) mlog()<<std::abs(qd.dot(q))<<std::endl;
        }
        return qd;
    }
    virtual Vector3d ts(double time, int derivative) const{
        if(derivative==0) return pose(time).t;
        return (ts(time+delta,derivative-1) -
                ts(time-delta,derivative-1))/(2.0*delta);
    }
    virtual Vector3d tws(double time, int derivative) const{
        if(derivative==0) return pose(time).getTinW();
        return (tws(time+delta,derivative-1) -
                tws(time-delta,derivative-1))/(2.0*delta);
    }

    PoseD pose(double time) const
    {
        PoseD P0=pose2(0);
        return P0.inverse()*pose2(time);
    }

    PoseD pose2(double time) const
    {

        time*=0.01;
        Vector3d center(0,0,0);
        auto tinw=[](double t){
            double w=2*3.1415;
            double w2=w/5.0;

            auto x0=Vector3d(std::cos(w*t),std::sin(w*t),std::cos(w2*t)*1.5)*10.0;
            auto x1=Vector3d(std::cos(w2*t)*1.5,std::sin(w*t),std::cos(w*t))*20;

            // 0 at -inf-0.8, then 1 at 1.2 to inf
            auto sigmoid=[](double x){                        
                x=x-1;
                x=x*15+3.1415;

                return 1.0/(1.0+std::exp(-x));
            };
            double a=sigmoid(t);

            return a*x0+(1.0-a)*x1;
        };
        PoseD P=lookAt(center,
                       tinw(time),
                       (tinw(time)-tinw(time-0.01))).inverse();
        P.t+=Vector3d(std::cos(time*0.1),-std::sin(time*0.1),std::cos(time*0.1))*1;

        return P;
    }

};





}
