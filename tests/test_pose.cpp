#include <iostream>
#include <vector>
#include "mlib/utils/cvl/pose.h"
#include <mlib/utils/cvl/rotation_helpers.h>
#include <mlib/utils/numerics.h>
#include <mlib/utils/constants.h>
#include <mlib/tests/datafile.h>
#include <mlib/utils/random.h>
#include <mlib/utils/simulator_helpers.h>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

using std::cout;using std::endl;
// rotation tests

using namespace cvl;
using namespace mlib;


void testGenerator()
{
    Matrix3d Rx=getRotationMatrixX(pi/2);
    CHECK(almost(Rx*Vector3d(1,2,3),Vector3d(1,-3,2)));
    Matrix3d Ry=getRotationMatrixY(pi/2);
    CHECK(almost(Ry*Vector3d(1,2,3),Vector3d(3,2,-1)));
    Matrix3d Rz=getRotationMatrixZ(pi/2);
    CHECK(almost(Rz*Vector3d(1,2,3),Vector3d(-2,1,3)));
    //cout<<"Rotation"<<Rx*Ry*Rz<<endl;// also ensures that the code is not optimized away
}


bool testRotationMatrix(Matrix3d R,std::vector<Vector3d> xs,std::vector<Vector3d> xrs){
    CHECK(xs.size()==xrs.size());
    CHECK(xs.size()>0);
    CHECK(R.isnormal());
    CHECK(isRotationMatrix(R));

    for(uint i=0;i<xs.size();++i){
        auto a=xrs[i];
        auto b=R*xs[i];

        CHECK((a-b).abs().sum() == doctest::Approx(0.0).epsilon(1e-11));

    }
    return true;
}
bool testRotationQuaternion(Vector4d Rq,std::vector<Vector3d> xs,std::vector<Vector3d> xrs){
    CHECK(xs.size()==xrs.size());
    CHECK(xs.size()>0);
    CHECK(Rq.isnormal());
    Rq.normalize();
    CHECK(isRotationQuaternion(Rq));

    for(uint i=0;i<xs.size();++i){
        Vector3d a=xrs[i];
        Vector3d b=quaternionRotate(&Rq[0],xs[i]);

        if((a-b).length()/3.0>1e-6){
            // cout<<"|a-b|: "<<(a-b).length()<<endl;
            return false;
        }
    }
    return true;
}
std::vector<Vector3d> rot(Matrix3d R,std::vector<Vector3d> xs){
    std::vector<Vector3d> ret;
    for(Vector3d x:xs)
        ret.push_back(R*x);
    return ret;
}
void testTr0RotationConversions(){
    //cout<<"Warning: I have not found a rotation matrix4x4 with a bad trace to test all the failure cases"<<endl;

    Matrix3d Rtr0(-0.99883423120358283, 0.048271423384232223, -0.00021968251702566221,
                  0.048224480889635574, 0.99804394664583307, 0.039782910986193187,
                  0.0021396305459731365, 0.039725939234590738, -0.99920832248988845);

    std::vector<Vector3d> trx=rot(Rtr0,x);// certain this works now since after other tests
    CHECK(testRotationQuaternion(getRotationQuaternion(Rtr0),x,trx));
}
bool rotationMatrixRotates(){
    // first test the generator
    //cout<<"Testing the rotation matrix generator";
    testGenerator();
    //cout<<" - Done"<<endl;
    // test with known rotations and vectors
    std::vector<Matrix3d> Rs;Rs.reserve(angles.size());
    for(const Vector3d& angle:angles)
        Rs.push_back(getRotationMatrixXYZ(angle));
    CHECK(xr.size()==Rs.size());
    //cout<<"Testing rotation by rotation matrix"<<endl;
    for(uint i=0;i<xr.size();++i){
        CHECK(testRotationMatrix(Rs[i],x,xr[i]));
    }
    //cout<<"Testing rotation by rotation Quaternion and matrix to quat";
    for(uint i=0;i<xr.size();++i){
        //Vector4 q=getRotationQuaternion(Rs[i]);
        // cout<<"R"<<Rs[i]<< "\n q:"<<q<<endl;
        // cout<<"R*x:   "<<Rs[i]*xr[i][0]<<"\nRq(x): "<<quaternionRotate(&q[0],xr[i][0])<<endl;
        CHECK(testRotationQuaternion(getRotationQuaternion(Rs[i]),x,xr[i]));
    }
    //cout<<" - Done"<<endl;

    testTr0RotationConversions();

    //cout<<"Testing quaternion to Matrix conversion";
    std::vector<cvl::Vector4d> qs;qs.reserve(Rs.size());
    for(auto& R:Rs){
        Matrix3d Rq=getRotationMatrix(getRotationQuaternion(R));
        CHECK(almost(R,Rq));
        Matrix3d Rq2=getRotationMatrix(getRotationQuaternion(getRotationMatrix(getRotationQuaternion(R))));
        CHECK(almost(R,Rq2));
        Matrix3d R3=getRotationMatrix(-getRotationQuaternion(R));
        CHECK(almost(R,R3));
    }
    //cout<<" - Done"<<endl;
    return true;
}

bool testQuaternionConversion(){
    //cout<<"Testing Rotation Matrix to Quaternion "<<endl;
    // identity is a good start:
    Matrix3d I=Matrix3d::Identity();
    Matrix3d I0=Matrix3d(1,0,0,
                             0,1,0,
                             0,0,1);

    CHECK(almost(I,I0));

    Vector4d q=getRotationQuaternion(I);
    CHECK(almost(q,Vector4d(1,0,0,0)));
    // other way around
    Matrix3d R0=getRotationMatrix(q);
    CHECK(almost(R0,I));

    //cout<<"Testing Rotation Matrix to Quaternion -done "<< endl;
    return true;
}
bool poseTransforms(){
    //cout<<"Testing pose transforms";
    std::vector<PoseD> ps;ps.reserve(angles.size());
    std::vector<Matrix3d> Rs;Rs.reserve(angles.size());
    for(const Vector3d& angle:angles)
        Rs.push_back(getRotationMatrixXYZ(angle));

    for(const Vector3d& angle:angles){
        ps.push_back(PoseD(getRotationMatrixXYZ(angle),angle));
    }


    CHECK(Rs.size()==ps.size());


    cout<<ps.size()<<endl;
    for(const Vector3d xi:x){
        for(uint i=0;i<ps.size();++i){
            {
                PoseD p=ps[i];
                Matrix3d R=Rs[i];
                Vector3d t=angles[i];
                CHECK((p*xi -  (R*xi + t)).abs().sum() ==doctest::Approx(0.0).epsilon(1e-11));
                CHECK(almost(p.inverse()*(p*xi),xi));
                Matrix3d Rt=Rs[i].transpose();
                Vector3d tt=-Rt*angles[i];
                CHECK(almost(p.inverse()*xi,Rt*xi + tt));
            }
        }
    }
    // check propagation
    {
        for(Vector3d xi: x){
            PoseD pc;
            Vector3d xc=xi;
            for(PoseD p:ps){
                pc=p*pc;
                xc=p*xc;
            }
            // low accuracy in something...
            CHECK((xc - (pc*xi)).abs().sum() == doctest::Approx(0.0).epsilon(1e-11));
        }
    }

    //cout<<" - Done"<<endl;
    return true;
}


double error(Matrix3d A, Matrix3d B){
    double e=0;
    for(int i=0;i<9;i++)
        e+=std::abs((A[i]-B[i]));
    return e;
}
double error(Vector4d A, Vector4d B){
    return std::min((A-B).length(),(A+B).length());
}

bool testConversions(){
    std::vector<Matrix3d> Rs;
    std::vector<Vector4d> qs;
    // matched pairs...

    // identity
    Rs.push_back(Matrix3d(1,0,0,0,1,0,0,0,1));
    qs.push_back(Vector4d(1,0,0,0)); // this is identity!
    // each axis separetly
    for(int i=1;i<2;++i){
        double a=i*pi_d/4.0;
        Rs.push_back(getRotationMatrixX(a));
        Rs.push_back(getRotationMatrixY(a));
        Rs.push_back(getRotationMatrixZ(a));
        for(int n=1;n<4;++n){

            double q0=std::cos(a/2.0);
            double qa=sqrt(1-q0*q0);
            Vector4d q(q0,0,0,0);
            q[n]=qa;
            qs.push_back(q);
        }
    }


    CHECK(Rs.size()==qs.size());
    for(uint i=0;i<Rs.size();++i){
        Matrix3d R = getRotationMatrix(qs[i]);
        Vector4d q   = getRotationQuaternion(Rs[i]);
        //cout<<"error: "<<error(R,Rs[i])<<" "<<error(q,qs[i])<<endl;
        //cout<<"R: \n"<<R<<endl;
        //cout<<"Rs[i]: \n"<<Rs[i]<<endl;
        CHECK(error(R,Rs[i])<1e-14);
        //cout<<"Q: "<<q<<endl;
        //cout<<"Qs[i]: "<<qs[i]<<endl<<endl<<endl;
        CHECK(error(q,qs[i])<1e-14);
    }
    return true;
}


















TEST_CASE("QUATERION,PRODUCT_BASIC"){
    Vector4d s(1,0,0,0);
    Vector4d i(0,1,0,0);
    Vector4d j(0,0,1,0);
    Vector4d k(0,0,0,1);


    {
        Vector4d r=QuaternionProduct(s,s);
        CHECK(almost(r[0],1)); // scalar
        CHECK(almost(r[1],0));
        CHECK(almost(r[2],0));
        CHECK(almost(r[3],0));
    }
    // e^2 =-1
    {
        Vector4d r=QuaternionProduct(i,i);
        CHECK(almost(r[0],-1));
        CHECK(almost(r[1],0));
        CHECK(almost(r[2],0));
        CHECK(almost(r[3],0));
    }
    {
        Vector4d r=QuaternionProduct(j,j);
        CHECK(almost(r[0],-1));
        CHECK(almost(r[1],0));
        CHECK(almost(r[2],0));
        CHECK(almost(r[3],0));
    }
    {
        Vector4d r=QuaternionProduct(k,k);
        CHECK(almost(r[0],-1));
        CHECK(almost(r[1],0));
        CHECK(almost(r[2],0));
        CHECK(almost(r[3],0));
    }
    // ijk=-1
    {
        Vector4d ij=QuaternionProduct(i,j);
        Vector4d ijk=QuaternionProduct(ij,k);
        CHECK(almost(ijk[0],-1));
        CHECK(almost(ijk[1],0));
        CHECK(almost(ijk[2],0));
        CHECK(almost(ijk[3],0));
    }
    {
        // ij=k
        Vector4d ij=QuaternionProduct(i,j);
        CHECK(almost(ij,k));
        // ji=-k
        Vector4d ji=QuaternionProduct(j,i);
        CHECK(almost(ji,-k));
        //jk=i
        Vector4d jk=QuaternionProduct(j,k);
        CHECK(almost(jk,i));
        //kj=-i
        Vector4d kj=QuaternionProduct(k,j);
        CHECK(almost(kj,-i));
        //ki=j
        Vector4d ki=QuaternionProduct(k,i);

        CHECK((ki - j).abs().sum() == doctest::Approx(0.0).epsilon(1e-11));
        //ik=i
        Vector4d ik=QuaternionProduct(i,k);
        CHECK(almost(ik,-j));
    }

    // inverse
    // inverse twice
    CHECK(almost(s+i+j+k , invertQuaternion(invertQuaternion(s+i+j+k))));

    return;

    {
        // quaternion rotation:
        Vector3d x(1,2,3);
        Vector3d xs(1,2,3);
        Vector3d xi(1,-2,-3);
        Vector3d xj(-1,2,-3);
        Vector3d xk(-1,-2,3);


        Matrix3d Rs=getRotationMatrix(s);     // angle is acos(1) = 0;        Axis is undefined
        Matrix3d Ri=getRotationMatrix(i);     // angle is acos(0) = pi;       Axis is (1,0,0)
        Matrix3d Rj=getRotationMatrix(j);     // angle is acos(0) = pi;       Axis is (0,1,0)
        Matrix3d Rk=getRotationMatrix(k);     // angle is acos(0) = pi;       Axis is (0,0,1)

        // checks that the transforms work
        CHECK(almost(Rs*x,xs));
        CHECK(almost(Ri*x,xi));
        CHECK(almost(Rj*x,xj));
        CHECK(almost(Rk*x,xk));



        CHECK(almost(QuaternionRotate(s,x),xs));
        CHECK(almost(QuaternionRotate(i,x),xi));
        CHECK(almost(QuaternionRotate(j,x),xj));
        CHECK(almost(QuaternionRotate(k,x),xk));

        Vector4d sijk=s+i+j+k;sijk.normalize();
        Matrix3d Rsijk=getRotationMatrix(sijk);
        Vector3d xsijk(3, 1, 2);
        CHECK(almost(Rsijk*x, xsijk));
        CHECK(almost(QuaternionRotate(sijk,x), xsijk));
    }
}

TEST_CASE("QUATERION,RANDOM"){

    int N=1000;
    std::vector<Vector3d> xs;xs.reserve(N);
    for(int i=0;i<N;++i)
        xs.push_back(getRandomUnitVector<double,3>()*randn(0,1));
    for(auto x:xs){
        PoseD p=getRandomPose();
        p.getTRef()[0]=0;
        p.getTRef()[1]=0;
        p.getTRef()[2]=0;
        Vector3d xr=cvl::QuaternionRotate(p.getQuaternion(),x);
        CHECK(almost(xr,p*x));
    }



    for(int i=0;i<N;++i){
        PoseD a=getRandomPose();
        PoseD b=getRandomPose();
        PoseD c=a*b;
        Vector4d pcq=c.getQuaternion();
        Vector4d cq= QuaternionProduct(a.getQuaternion(),b.getQuaternion());
        if(cq[0]<0) cq=-cq;
        if(pcq[0]<0) pcq=-pcq;
        CHECK((pcq-cq).abs().sum() == doctest::Approx(0.0).epsilon(1e-11));
    }

}



TEST_CASE("POSE,QUATERIONCONVERSION"){
    CHECK(testQuaternionConversion());
    CHECK(rotationMatrixRotates());
    CHECK(poseTransforms());
    CHECK(testConversions());
}


/*
TEST(CONVERSIONS_CVL2EIGEN,Pose2Isometry){
    PoseD P;
    Eigen::Transform<double,3,Eigen::Isometry> m=convert2isometry(P);
    for(int i=0;i<4;++i)
        for(int j=0;j<4;++j)
            if(i==j)
                CHECK(m(i,j),1,1e-13);
            else
                CHECK(m(i,j),0,1e-13);
}

TEST(CONVERSIONS_CVL2EIGEN,Isometry2Pose){

    Eigen::Transform<double,3,Eigen::Isometry> m;
    for(int i=0;i<4;++i)
        for(int j=0;j<4;++j)
            if(i==j)
                m(i,j)=1;
            else
                m(i,j)=0;
    PoseD P=convert2cvl(m);
    Vector4d q=P.getQuaternion();
    CHECK(q.length(),1,1e-12);
    CHECK(P.getT().length(),0,1e-12);
}

Vector3d apply(Eigen::Transform<double,3,Eigen::Isometry> iso,Vector4d x){
    Vector3d r(0,0,0);
    for(int i=0;i<3;++i)
        for(int j=0;j<4;++j)
            r(i)+=iso(i,j)*x(j);
    return r;
}

void SimilarTransform(PoseD P, Eigen::Transform<double,3,Eigen::Isometry> iso){
    std::vector<Vector3d> xs;
    xs.push_back(Vector3d(5,2,3));
    xs.push_back(Vector3d(2,5,3));
    xs.push_back(Vector3d(5,3,2));
    xs.push_back(Vector3d(0,0,0));
    for(uint i=0;i<xs.size();++i){
        Vector3d x=apply(iso,xs[i].homogeneous());
        CHECK(almost(x,P*xs[i]));
    }
}

TEST(CONVERSIONS_CVL2EIGEN,POSE2ISO){
    PoseD P=getRandomPose();
    Eigen::Transform<double,3,Eigen::Isometry> iso=convert2isometry(P);
    SimilarTransform(P,iso);
}

*/

TEST_CASE("POSE_LOOK_AT,BASIC"){
    cvl::Vector3d point(0,0,0);
    cvl::Vector3d from(1,2,3);
    PoseD P=cvl::lookAt(cvl::Vector3d(0,0,0),from,cvl::Vector3d(0,1,0));
    cvl::Vector3d tw=P.getTinW();
    CHECK(almost(tw,from));

    cvl::Vector3d x=cvl::Vector3d(0,0,0);
    x=P*x;
    CHECK(almost(x[0],0));
    CHECK(almost(x[1],0));
    CHECK(almost(x[2],std::sqrt(1+4+9)));

    x=cvl::Vector3d(1,2,3);
    auto y=Vector3d(0,0,0);
    x=P*x;

    CHECK(almost(x,y));

}




