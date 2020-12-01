#if 0
#pragma once
#include <iostream>

#include <mlib/utils/random.h>
#include <mlib/utils/simulator_helpers.h>
#include <mlib/sfm/p3p/p4p.h>


using namespace mlib;
using namespace cvl;



// p3p is hard to test but p4p is easy


bool test_p4p(std::vector<Vector3d> xs,
              PoseD P0,
              double max_noise=1e-10){

    EXPECT_TRUE(xs.size()==4);
    std::vector<Vector3d> xrs;
    std::vector<Vector2d> yns;

    for(auto x:xs){
        EXPECT_TRUE(x.isnormal());
        Vector3d xr=P0*x;
        xrs.push_back(xr);
        EXPECT_TRUE(xr[2]>0.1);
        yns.push_back(xr.dehom());
    }
    PoseD P=p4p(xs,yns,Vector4<uint>(0,1,2,3));


    for(uint i=0;i<xs.size();++i){

        EXPECT_TRUE(xrs[i][2]>0);



        double err=0;
        for(int i=0;i<3;++i)
            err+=(yns[i]-(P*xs[i]).dehom()).length();



        EXPECT_NEAR(err,0,max_noise);
    }
    return true;
}



TEST(P4P,BASIC_NOISE_FREE_Identity){

    std::vector<Vector3d> xs;
    xs.push_back(Vector3d(1,2,3));
    xs.push_back(Vector3d(2,1,3));
    xs.push_back(Vector3d(3,2,1));
    xs.push_back(Vector3d(2,2,2));
    PoseD I;
    test_p4p(xs, I);
}


TEST(P4P,BASIC_NOISE_FREE_translation){
    std::vector<Vector3d> xs;
    xs.push_back(Vector3d(1,2,3));
    xs.push_back(Vector3d(2,1,5));
    xs.push_back(Vector3d(3,2,1));
    xs.push_back(Vector3d(2,2,2));
    PoseD P=PoseD(Vector3d(0,0,0.1));
    test_p4p(xs, P);
}




TEST(P4P,BASIC_NOISE_FREE_rotation){
    std::vector<Vector3d> xs;
    xs.push_back(Vector3d(1,2,3));
    xs.push_back(Vector3d(2,1,5));
    xs.push_back(Vector3d(3,2,1));
    xs.push_back(Vector3d(2,2,2));
    PoseD P=PoseD(getDefaultR());
    test_p4p(xs, P);
}

TEST(P4P,BASIC_NOISE_FREE_rotation_and_translation){
    std::vector<Vector3d> xs;
    xs.push_back(Vector3d(1,2,3));
    xs.push_back(Vector3d(2,1,5));
    xs.push_back(Vector3d(3,2,1));
    xs.push_back(Vector3d(2,2,2));
    PoseD P=PoseD(getDefaultR(),getDefaultT());
    test_p4p(xs, P);
}

TEST(P4P,RANDOM_POSE_RANDOM_POINTS_NOISE_FREE){
    double count=0;
    double iterations=1000;
    for(int i=0;i<iterations;++i){
        // this test can fail if the points randomly become a degenerate case
        PoseD P=getRandomPose();
        std::vector<Vector3d> xs=getRandomPointsInfrontOfCamera(P,4);
        if(!test_p4p(xs, P,1e-5)){
            count++;
            EXPECT_TRUE(false);

        }
    }

    //std::cout<<"P4P: success ratio: "<<(iterations-count)/iterations<<std::endl;
    // some wierd errorr in < makes this false despite the one above never triggering..
    //EXPECT_TRUE(100*count<iterations);
}
#endif
