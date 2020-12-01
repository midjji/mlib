
/**
 * This file tests the different types of triangulation
 * Assumes Poses are tested(Matrix,Rotation), Assumes Random is tested
 *
 *
 * Several types of triangulation are available
 *
 * StereoTriangulation ie just with a baseline
 *
 *
 *
 *
 */






#include <mlib/utils/random.h>
#include <mlib/utils/simulator_helpers.h>
#include <mlib/utils/cvl/triangulate.h>
#include <mlib/ceres_util/triangulation.h>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
using namespace mlib;
using namespace cvl;

using std::cout;using std::endl;


TEST_CASE("TRIANGULATION_STEREO, Basic"){
    // Trivial example, no numerics to cloud the issue
    Vector3d x={1,2,3};
    double baseline = 1;
    Vector3d xr = PoseD(-Vector3d(baseline, 0, 0)) * x; // second camera on the right=> -baseline
    Vector3d res = triangulate(x.dehom(), xr.dehom(), baseline);
    Vector2d dx = x.dehom() - xr.dehom(); // disparity should be positive

    // disparity is from right to left
    CHECK(dx(0)>0);
    // no change in y coordinate
    CHECK(dx(1) == doctest::Approx(0.0).epsilon(1e-13));
    // reconstruction should be good
    CHECK((res-x).length()/x.length()==doctest::Approx(0.0).epsilon(1e-13));
}

TEST_CASE("TRIANGULATION_STEREO, RANDOM_EASY"){
    // a very nice set of random data that should always succeed

    for(int i=0;i<1000;++i) {
        Vector3d x = getRandomUnitVector<double,3>();
        x[2] = std::abs(x[2]) + 1.1;
        //x={1,2,3};

        double baseline = randu<double>(0.5, 1.5);
        CHECK(baseline>0.4);
        //baseline = 1;
        Vector3d xr = PoseD(-Vector3d(baseline, 0, 0)) * x; // second camera on the right=> -baseline

        Vector3d res = triangulate(x.dehom(), xr.dehom(), baseline);
        auto dx = x.dehom() - xr.dehom(); // disparity should be positive

        // disparity is from right to left
        CHECK(dx(0)>0);
        // no change in y coordinate
        CHECK(dx(1)==doctest::Approx(0.0).epsilon(1e-13));

        CHECK((res-x).length()/x.length()==doctest::Approx(0.0).epsilon(1e-13));

    }
}




// midpoint general triangulation tests





/**
 * @brief basicTest
 * triangulation is always correct even to scale if the pose is known!
 * tests some very basic variants to ensure the directions are right etc...
 */
TEST_CASE("TRIANGULATION_MIDPOINT, BASIC") {

    Vector3d x(0,0,10);
    PoseD Pqp(Vector3d(1,0,0));
    Vector2d pn=x.dehom();
    Vector3d xq=Pqp*x;
    Vector2d qn=xq.dehom();

    auto xr=triangulate(PoseD(),Pqp,pn,qn);
    CHECK((x-xr).length()==doctest::Approx(0.0).epsilon(1e-13));
}

void gentest(Vector3d x_world, PoseD Ppw, PoseD Pqw){
    Vector3d xp=Ppw*x_world;
    Vector3d xq=Pqw*x_world;
    CHECK(xp(2)>0);
    CHECK(xq(2)>0);

    Vector2d pn=xp.dehom();
    Vector2d qn=xq.dehom();

    Vector3d xr=triangulate(Ppw,Pqw,pn,qn);
    // triangulation is num unstable for some configurations
    CHECK((xr-x_world).length()==doctest::Approx(0.0).epsilon(1e-13));


    // now the relative transf ...
    PoseD Pqp=Pqw*(Ppw.inverse());
    xr=triangulate(PoseD(),Pqp,pn,qn);

    CHECK((xr-Ppw*x_world).length()==doctest::Approx(0.0).epsilon(1e-13));
}


TEST_CASE("TRIANGULATION_MIDPOINT,RANDOM_EASY"){
    gentest(Vector3d(1,2,3),PoseD(),PoseD(Vector3d(-0.38,0,0)));
    for(int i=0;i<1000;++i)
    {
        Vector3d x_world(randu<double>(-1,1),randu<double>(-1,1),randu<double>(-1,1));
        PoseD Ppw=getRandomPose();
        PoseD Pqw=getRandomPose();
        Vector3d xp=Ppw*x_world;
        Vector3d xq=Pqw*x_world;
        if(xp(2)<0.1 or xq(2)<0.1)
            continue;

        gentest(x_world,Ppw,Pqw);
    }
}



