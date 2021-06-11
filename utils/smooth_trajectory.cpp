#include <mlib/utils/smooth_trajectory.h>
#include <mlib/utils/cvl/polynomial.h>
#include <mlib/utils/cvl/lookat.h>
#include <mlib/utils/smooth_trajectory_axis_loop.h>
#include <mlib/utils/random.h>
using std::cout;
using std::endl;
namespace cvl{

PoseD SmoothTrajectory::operator()(double time) const{
    return pose(time);
}
PoseD SmoothTrajectory::pose(double time) const
{



    // that would mess up the derivatives!
    return PoseD(this->qs(time, 0), this->ts(time,0));

    //PoseD P0=this->pose2((10-t0())/(t1()-t0()));
    //return P0.inverse()*this->pose2(t);
}


std::vector<PoseD> SmoothTrajectory::display_poses(std::vector<double> ts) const{
    cout<<"display poses ts: "<<ts.size()<<endl;
    std::vector<PoseD> ps;ps.reserve(ts.size());
    // we assume its in x_world=P(t)x_camera, so invert for display?
    for(auto t:ts)
        ps.push_back(pose(t).inverse());
    return ps;
}


std::vector<double> SmoothTrajectory::interior_times(
        int samples_per_second,
        int border) const{
    int N=t1()-t0();N*=samples_per_second;
    std::vector<double> ts;ts.reserve(N);

    double d=(t1()-t0())/double(N);
    for(int i=border;i<N-border;++i){
        double time=t0()+d*i;
        ts.push_back(time);
    }
    return ts;
}





std::vector<std::shared_ptr<SmoothTrajectory>> test_trajectories(){
    std::vector<std::shared_ptr<SmoothTrajectory>> tests;
    tests.push_back(std::make_shared<AxisLoop>(Vector3d(1,0,0),1));
    tests.push_back(std::make_shared<AxisLoop>(Vector3d(0,1,0),1));
    tests.push_back(std::make_shared<AxisLoop>(Vector3d(0,0,1),1));
    tests.push_back(std::make_shared<AxisLoop>(Vector3d(0,0,1),0.1));
    auto random_axis=[](){Vector3d axis(0,0,0);
        while(axis.norm()==0){
            axis=Vector3d(mlib::randn(),mlib::randn(),mlib::randn());
        }
        return axis.normalized();
    };
    tests.reserve(3);
    for(int i=0;i<3;++i)
            tests.push_back(std::make_shared<AxisLoop>(random_axis(),0.1));

    return tests;
}




#if 0
Vector4d SmoothTrajectory::qs(double time, int derivative) const{
    if(derivative==0) return pose(time).q();

    auto a=qs(time+delta,derivative-1);
    auto b=qs(time-delta,derivative-1);
    // this means we are currently rotating from b to a at a time of 2delta.
    // so if my q math is trust worthy, then q(t) \approx b(b^ca)^{2deltat}?
    // from which I can compute both omega etc...but maybe compare to this?
    Vector4d qd= a-b;

    if((a+b).norm()<(a-b).norm()) qd=a+b; // should I do this?
    qd=qd/(2.0*delta);
    if(derivative==1){
        Vector4d q = pose(time).q();
        // make it orthogonal to q
        qd=qd - qd.dot(q)*q;
    }
    return qd;
}
Vector3d SmoothTrajectory::ts(double time, int derivative) const{
    if(derivative==0) return pose(time).t();
    return (ts(time+delta,derivative-1) -
            ts(time-delta,derivative-1))/(2.0*delta);
}

PoseD AllRot::pose2(double time) const
{

    time*=0.01;
    Vector3d center(0,0,0);
    auto tinw=[](double t){
        double w=2*3.1415;
        double w2=5*w/5.0;

        auto x0=Vector3d(std::cos(w*t),std::sin(w*t),std::cos(w2*t)*1.5)*10.0;
        auto x1=Vector3d(std::cos(w2*t)*1.5,std::sin(w*t),std::cos(w*t))*20;

        // 0 at -inf-0.8, then 1 at 1.2 to inf
        auto sigmoid=[](double x){
            x=x-1;
            x=x*50+3.1415;

            return 1.0/(1.0+std::exp(-x));
        };
        double a=sigmoid(t);

        return a*x0+(1.0-a)*x1;
    };
    PoseD P=lookAt(center,
                   tinw(time),
                   (tinw(time)-tinw(time-0.01))).inverse();


    P.set_t(P.t()+Vector3d(std::cos(time*0.1),-std::sin(time*0.1),std::cos(time*0.1))*1);

    return P;
}








template<int degree, class Type=long double>
/**
 * @brief get_spline_basis_polys
 * @return
 *
 * This is the basis polynomial
 *
 * it is tested!
 */
CompoundBoundedPolynomial<degree,Type> get_spline_basis_polys() {
    CompoundBoundedPolynomial<degree,Type> ret;
    if constexpr (degree==0){
        ret.add(BoundedPolynomial<0,Type>(Vector2<long double>(0,1),Polynomial<0,Type>(1)));

    }
    else
    {
        long double k=1.0/((long double)degree);
        CompoundBoundedPolynomial<degree,Type> a=
                get_spline_basis_polys<degree-1,Type>()*
                (Polynomial<1,Type>(0,1)*Type(k));

        CompoundBoundedPolynomial<degree,Type> b=
                get_spline_basis_polys<degree-1,Type>().reparam(-1)*
                (Polynomial<1,Type>(degree+1,-1)*Type(k));

        for(auto bp:a.polys)ret.add(bp);
        for(auto bp:b.polys)ret.add(bp);

    }
    ret.collapse();
    return ret;
}

template<int degree>

CompoundBoundedPolynomial<degree>
get_spline_cumulative_basis_polys()
{ // this
    CompoundBoundedPolynomial<degree> ret;
    for(int j=0;j<degree+1 ;++j) {
        auto b=get_spline_basis_polys<degree>().reparam(-j);
        for(auto p:b.polys) ret.add(p);
    }
    ret.collapse();
    ret.bound(Vector2<long double>(0,degree));
    ret.add(BoundedPolynomial<degree>({degree,std::numeric_limits<long double>::max()},1));
    ret.collapse();
    return ret;
}




namespace  {
CompoundBoundedPolynomial<5> interpolator=get_spline_cumulative_basis_polys<5>();
}

double smooth_interpolator(double t){    
    return interpolator(t*5 -5);
}
double AllRot2::s(double t) const{return smooth_interpolator(t);}


Vector3d AllRot2::p(double time) const{    
    double t=8.0*(time-t0())/(t1()-t0());


    // total length of curve is,
    // from 0 to 8.
    // first rotate around origin at distance 1
    // the interpolate to axis 2,
    // then around axis 2
    // the interpolate to axis 3,
    // then around axis 3
    // then around axis 3 at 2x speed, // linear increase
    // then around axis 3 at 4x speed,


    double pi=3.14159265359;
    double w=2.0*pi;




    // its simple enough to make it continuous, making its derivatives continuous is harder.
    Vector3d z(std::cos(w*t),std::sin(w*t),0); // rotate around z axis
    Vector3d x(0,std::cos(w*t),std::sin(w*t)); // rotate around x axis


    // accellerating rotation around the y axis,
    // also drop it in y a little bit after t==.
    double f=2;
    Vector3d y(std::cos(w*(t + f*s(t-4)*(t-5))),0.1*s(t-4)*(t-5),std::sin(w*(t + f*s(t-4)*(t-5)))); // rotate around y axis




    Vector3d center(0,0,0);

    center= z*(1.0-s(t)) + s(t)*(x*(1.0-s(t-2)) + s(t-2)*y);
    return center*10;
}


PoseD AllRot2::pose2(double time) const
{
    // we are making two assumptions,
    // one,
    if(p(time).norm()<0.1){
        mlog()<<"bad... \n";
    }
    // two
    double dt=1e-4*(t1()-t0());
    if((p(time+dt)-p(time-dt)).norm()<1e-4){
        mlog()<<"bad... "<<(p(time+dt)-p(time-dt)).norm()<<"\n";
    }


    PoseD P=lookAt(Vector3d(0,0,0),
                   p(time),
                   (p(time+dt)  - p(time-dt)).normalized()).inverse();


    return P;
}




Vector3d AllRot3::position(double time) const{
    double t=(time-t0())/(t1()-t0());
    Vector3d p(1,0,0);
    double pi=3.14159265359*0.25;
    return (getRotationMatrixXYZ(2*pi*t,8*pi*t,32*pi*t)*p)*10.0;
}

PoseD AllRot3::pose2(double time) const
{
    double dt=1;

    PoseD P=lookAt(Vector3d(0,0,0),
                   position(time),
                   (position(time+dt)  - position(time-dt)).normalized()).inverse();


    return P;
}




#endif

}
