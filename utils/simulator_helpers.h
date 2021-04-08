#pragma once
/* ********************************* FILE ************************************/
/** \file    simulator_helpers.h
 *
 * \brief    This header contains various functions and classes that simplify generating random data pointcloud and pointcloud projection data for testing sfm components
 *
 * \remark
 * - c++11
 * - no dependencies
 * - header only
 *
 *
 *
 *
 *
 * \author   Mikael Persson
 * \date     2010-04-01
 * \note MIT licence
 *
 ******************************************************************************/

#include <math.h>
#include "mlib/utils/cvl/pose.h"
#include "mlib/utils/random.h"
#include <mlib/utils/random_vectors.h>


namespace cvl{

Matrix3d getRandomRotationInFirstOctant();
template<class T> Vector4<T> getRandomRotationQuaternion(double angle/* in radians as always*/){

    double q0=std::cos(angle/2);
    Vector3d n=mlib::getRandomUnitVector<double,3>();
    n*=std::sqrt(1-q0*q0);

    Vector4<T> q(q0,n(0),n(1),n(2));
    assert(fabs(q.length()-1)<1e-6);
    q.normalize();
    return q;
}

template<class T>
Vector4<T> getRandomRotation2(T angle_min, T angle_max){
    Vector3<T> n=mlib::getRandomUnitVector<T,3>();
    T angle=mlib::randu(angle_min,angle_max);
    T q0=cos(angle/2);
    T s=std::sqrt(1-q0*q0);
    n*=s;
    Vector4<T> q(q0,n[0],n[1],n[2]);
    return q;
}
// not uniform but guaranteed to be less than a tenth of a rad
Matrix3d getSmallRandomRotation();


// parameters of the default simulation
Matrix3d getDefaultR();
Vector3d getDefaultT();
Matrix3d getDefaultK();
std::vector<double> getDefaultDistortionParams();



// random geom
Vector3d getRandomPointOnPlane(const  Vector4d& n);
PoseD getRandomPose();
std::vector<Vector3d> getRandomPointCloud(uint size=500);
Matrix3d getDefaultIntrinsics();
Vector2d getDefaultImageSize();











/**
 * @brief getRandomPointsInfrontOfCamera
 * @param Pcw  x_cam=P*x_world
 * @param N number of points
 * @param minImageDistance minimum relative distance, might be ignored if the value is too low
 * @return x_world
 *
 * point vector where each point is a minimum distance from the rest
 *  points are random between -1,1 in x,y
 *
 */






std::vector<Vector3d> getRandomPointsInfrontOfCamera(PoseD Pcw,
                                                          uint N);

Vector3d getRandomPointInfrontOfCamera(PoseD Pcw);


std::vector<cvl::Vector3d> getRandomPointsInfrontOfTwoCamera(cvl::PoseD Pc1w, cvl::PoseD Pc2w,
                                                          uint N);











/**
 * @brief The PointCloudWithNoisyMeasurements class
 * Create one and get 500 ish random points infront
 *  of a stereo camera pair at a random world position
 * the points are noisy
 * the points have a outlier ratio
 *
 * the pose is a random rotation and a random unit length translation
 *
 * the focal length is assumed 1000 => 0.001*sigma => sigma in pixels
 * the camera is 4 Mpixel and goes from -1,1
 *
 * all points have disparity, this disparity can have a error
 * the disparity error is 1D
 * all observed points are infront of both cameras
 * all observed points are a minimum of 0.01 away from each other.
 * the disparity has errors and outliers
 *
 * Prl is the camera on the right if the main one
 *
 * y=Pcw*x
 * yr=Prl*Pcw*x
 *
 * Outliers are y+u*randu(..) where u is a random unit vector
 *
 * Good for testing pnp & pnpd
 *
 */
class PointCloudWithNoisyMeasurements{
public:
    PointCloudWithNoisyMeasurements(uint N,double pixel_sigma,double outlier_ratio);


    std::vector<Vector3d> xs;
    std::vector<Vector2d> yns,yns_gt;
    PoseD Pcw;
};

/**
 * @brief The MultipleCamerasObservingOnePointCloud class
 * This class generates a pointcloud and a bunch of randomly placed and oriented cameras observing it
 * Essentially random but all cameras will observe the majority of the points.
 * The resulting measurements will test but not strain any method applied to them
 */
class MultipleCamerasObservingOnePointCloud{
public:
    void init(int cameras);
    std::vector<PoseD> Pcws;
    std::vector<Vector3d> xs;
    std::vector<std::vector<Vector2d>> ynss;

};


/**
 * @brief The MultipleCamerasObservingOnePointCloud class
 * This class generates a pointcloud and a bunch of randomly placed and oriented cameras observing it
 * Essentially random but all cameras will observe the majority of the points.
 * The resulting measurements will test but not strain any method applied to them
 */
class NMovingCamerasObservingOnePointCloud{
public:
    void init(int cameras);
    std::vector<PoseD> Pcws;
    std::vector<Vector3d> xs;
    std::vector<std::vector<Vector2d>> ynss;
};

}// end namespace mlib
