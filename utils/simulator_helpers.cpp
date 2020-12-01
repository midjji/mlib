#include "mlib/utils/simulator_helpers.h"
#include "mlib/utils/random.h"
#include "mlib/utils/cvl/rotation_helpers.h"
#include "mlib/utils/constants.h"

#include <set>

#include <mlib/sfm/anms/grid.h>


using std::endl;
using std::cout;
using mlib::getRandomUnitVector;
using mlib::randn;
using mlib::randu;
using mlib::randui;
namespace cvl{


Matrix3d getRandomRotation(){
    Vector4d q=getRandomUnitVector<double,4>();
    return getRotationMatrix(q);
}

Matrix3d getRandomRotationInFirstOctant(){
    Vector4d q=getRandomUnitVector<double,4>();
    while(std::acos(q[0])>pi_d/8.0){
        q=getRandomUnitVector<double,4>();
    }

    Matrix3d R=getRotationMatrix(q);
    return R;
}

PoseD getRandomPose(){

    return PoseD(getRandomUnitVector<double,4>(),getRandomUnitVector<double,3>()*randn<double>(0.1,10));
}
std::vector<Vector3d> getRandomPointCloud(uint size){
    std::vector<Vector3d> xs;xs.reserve(500);
    for(uint i=0;i<size;++i)
        xs.push_back(Vector3d(randu<double>(-1,1),randu<double>(-1,1),randu<double>(-1,1)));
    return xs;
}
Matrix3d getDefaultIntrinsics(){
    return Matrix3d(800,0,400,
                    0,600,300,
                    0,0,1);
}
Vector2d getDefaultImageSize(){return Vector2d(800,600);}


Matrix3d getSmallRandomRotation(){

    Vector3d v;
    Vector3d axis(1,1,1);
    axis.normalize();
    while(true){
        v[0]=cap(randn<double>(0,1),-1,1);
        v[1]=cap(randn<double>(0,1),-1,1);
        v[2]=cap(randn<double>(0,1),-1,1);
        v.normalize();
        if(std::acos(v[0]*axis[0] + v[1]*axis[1] +v[2]*axis[2])<pi_d/4.0)
            break;
    }
    double th=randu<double>(0,2*pi_d);
    Matrix3d I,ux,uxu;
    I=Matrix3d(1,0,0,
               0,1,0,
               0,0,1);
    ux= v.crossMatrix();
    uxu=v*v.transpose();

    Matrix3d R=I*std::cos(th)+std::sin(th)*ux + (1-std::cos(th))*uxu;
    return R;
}



Matrix3d getDefaultR(){
    Matrix3d R;
    double th=0.1*pi_d;   // simple small rotation around z
    R(0,0)=cos(th);      R(0,1)=-sin(th);     R(0,2)=0;
    R(1,0)=sin(th);      R(1,1)=cos(th);      R(1,2)=0;
    R(2,0)=0;            R(2,1)=0;            R(2,2)=1;
    return R;
}
Vector3d getDefaultT(){
    return  Vector3d(1,0,1);
}
Matrix3d getDefaultK(){
    Matrix3d K(5.5297423155370348e+02,  0                     , 3.3038141055640062e+02,
               0                     ,  5.5297423155370348e+02, 2.5446246888992496e+02,
               0                     ,  0                     , 1                      );
    return K;
}
std::vector<double> getDefaultDistortionParams(){
    std::vector<double> de;
    de.push_back(-3.6842883904708343e-02);
    de.push_back(3.1108147363716358e-02);
    de.push_back(-1.9908199658443278e-04);
    de.push_back(-1.5572181673667229e-03);
    de.push_back(7.6694383986063527e-02);
    return de;
}




Vector3d getRandomPointOnPlane(const  Vector4d& n){
    assert(n.isnormal());
    assert(n.length()>1e-10);
    Vector4d N=n;
    N.normalize();
    assert(N.isnormal());
    double x,y,z;
    x=500*randu<double>(-1,1);
    y=500*randu<double>(-1,1);
    z=500*randu<double>(-1,1);

    if(fabs(N[0])>1e-6){
        x=(N[3] - N[2]*z - N[1]*y)/N[0];
    }
    if(fabs(N[0])>1e-6){
        y=(N[3] - N[2]*z - N[0]*x)/N[1];
    }
    if(fabs(N[2])>1e-6){
        z=(N[3] - N[0]*x - N[1]*y)/N[2];
    }
    return  Vector3d(x,y,z);
}




PointCloudWithNoisyMeasurements::PointCloudWithNoisyMeasurements(uint N,double pixel_sigma,double outlier_ratio){
    // generate 500 points infront of the camera

    Pcw=PoseD(getRandomRotation(),getRandomUnitVector<double,3>());
    xs=getRandomPointsInfrontOfCamera(Pcw,N);
    yns_gt.reserve(xs.size());
    yns.reserve(yns_gt.size());

    for(auto x:xs){
        auto y=(Pcw*x).dehom();
        yns_gt.push_back(y);
    }



    double sigma=pixel_sigma*0.001;

    for(Vector2d y:yns_gt){
        yns.push_back(y+getRandomUnitVector<double,2>()*sigma);
    }


    if(outlier_ratio>0){
        uint outliers=uint(xs.size()*outlier_ratio);
        for(uint i=0;i<outliers;++i){
            // exact ratio is not needed
            uint index=randui<uint>(0,yns.size()-1);

            auto y=yns[index];
            if(randu<double>(0,1)>0.5) // and something to really mess with things that dont cut
                y+=getRandomUnitVector<double,2>();
            while(((Pcw*xs[index]).dehom() - y).norm() < 0.002 + pixel_sigma*0.001*3)
                y+=getRandomUnitVector<double,2>()*0.1*randu<double>(3,10);
            yns.at(index)=y; // near but not close enough for a missleading match
        }
    }




}


void MultipleCamerasObservingOnePointCloud::init(int cameras){

    {// ball with cameras on the surface, all are facing inwards with random up vector and a small rotation error
        //ball has a radius of 10
        for(int i=0;i<cameras;++i){
            Vector3d t=getRandomUnitVector<double,3>()*10.0;
            // they should look in roughly the right direction but not exactly
            Vector3d center=getRandomUnitVector<double,3>();
            Vector3d up=getRandomUnitVector<double,3>();
            PoseD P=lookAt(center,t,up);
            Pcws.push_back(P);
        }
    }
    {// init the points and measurements
        while(xs.size()<500){
            Vector3d x(randu<double>(-9,9),randu<double>(-9,9),randu<double>(-9,9));

            std::vector<Vector2d> yns;
            for(PoseD P:Pcws){
                Vector3d xr=P*x;
                if(xr[2]<0.01) continue;
                Vector2d yn(xr[0]/xr[2],xr[1]/xr[2]);
                if(!yn.is_in({-1.0,-1.0},{1.0,1.0})) continue;
                yns.push_back(yn);
            }
            if(yns.size()!=Pcws.size()) continue;
            xs.push_back(x);
            ynss.push_back(yns);
        }
    }
}
void NMovingCamerasObservingOnePointCloud::init(int cameras){

    // camera 0 in identity
    // camera 1... offset with random unit translation and small rotation

    {// camera 0 on the surface of ball, the unit distance away
        //ball has a radius of 10
        Vector3d t=getRandomUnitVector<double,3>()*10.0;

        for(int i=1;i<cameras;++i){
            // they should look in roughly the right direction but not exactly
            Vector3d center=getRandomUnitVector<double,3>();
            Vector3d up=getRandomUnitVector<double,3>();
            PoseD P=lookAt(center,t,up);
            Pcws.push_back(P);
        }
    }
    {// init the points and measurements
        while(xs.size()<500){
            Vector3d x(randu<double>(-9,9),randu<double>(-9,9),randu<double>(-9,9));

            std::vector<Vector2d> yns;
            for(PoseD P:Pcws){
                Vector3d xr=P*x;
                if(xr[2]<0.01) continue;
                Vector2d yn(xr[0]/xr[2],xr[1]/xr[2]);
                if(!yn.is_in({-1.0,-1.0},{1.0,1.0})) continue;
                yns.push_back(yn);
            }
            if(yns.size()!=Pcws.size()) continue;
            xs.push_back(x);
            ynss.push_back(yns);
        }
    }
}


cvl::Vector3d getRandomPointInfrontOfCamera(cvl::PoseD Pcw){
    // reduce the odds that it just cycles!
    PoseD Pwc=Pcw.inverse();
    Vector2d yn=Vector2d(randu<double>(-1,1),randu<double>(-1,1));
    double distance =randu<double>(0.1,100); // about 0.1 to 1000 m(if 0,7)
    Vector3d xc=(yn.homogeneous()*distance);
    return Pwc*xc;
}
std::vector<Vector3d> getRandomPointsInfrontOfCamera(PoseD Pcw,
                                                     uint N) {
    // reduce the odds that it just cycles!
    std::vector<Vector3d> xs;xs.reserve(N);
    PoseD Pwc=Pcw.inverse();
    while(xs.size()<N){
        Vector2d yn=Vector2d(randu<double>(-1,1),randu<double>(-1,1));
        double distance =randu<double>(0.1,100); // about 0.1 to 1000 m(if 0,7)
        Vector3d xc=(yn.homogeneous()*distance);
        xs.push_back(Pwc*xc);
    }
    assert(xs.size()==N);
    return xs;
}

std::vector<Vector3d> getRandomPointsInfrontOfTwoCameras(PoseD Pc1w, PoseD Pc2w, uint N) {
    // reduce the odds that it just cycles!
    std::vector<Vector3d> xs;xs.reserve(N);
    PoseD Pwc1=Pc1w.inverse();
    while(xs.size()<N){
        Vector2d yn=Vector2d(randu<double>(-1,1),randu<double>(-1,1));
        double distance =randu<double>(0.1,100); // about 0.1 to 1000 m(if 0,7)
        Vector3d xc=(yn.homogeneous()*distance);
        Vector3d xw=Pwc1*xc;
        Vector2d yn2=(Pc2w*xw).dehom();
        if (yn2.absMax() > 1)
            continue;
        xs.push_back(xw);

    }
    assert(xs.size()==N);
    return xs;
}

}// end namespace mlib
