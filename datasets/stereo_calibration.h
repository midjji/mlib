#pragma once

/* ********************************* FILE ************************************/
/** \file    calibration.h
 *
 * \brief    The calibration
 *
 * \remark
 * - c++11
 *
 * \todo
 *
 *
 *
 * \author   Mikael Persson
 * \date     2019-01-01
 * \note GPL licence
 *
 ******************************************************************************/
#include <iostream>
#include <sstream>
#include <mlib/utils/cvl/pose.h>
#include <mlib/utils/cvl/triangulate.h>

namespace cvl{
/**
 * @brief The StereoCalibration class
 * Vector3 X: is 3d point with x,y,z right handed with x to the right, y down, and z forwards.
 * Vector4 X: is homogeneous 3d point i.e. Vector3 X_3= Vector4 X/Vector4 X[3]
 *
 * Vector2 y=project is image row, col
 * Vector3 y=project is image row, col, disparity
 *
 * disparity is Left(x) =Right(x-disparity)
 *
 * //TODO Add right disparity
 *
 *
 */
class StereoCalibration
{
public:



    int rows_=1024;
    int cols_=2048;

    double fy_=2261.54;
    double fx_=2267.22;
    double py_=1108.15;
    double px_=519.169;
    double baseline_=0.209114;
    PoseD P_left_vehicle_=PoseD::Identity(); // P_left_vehicle_
    inline PoseD P_left_vehicle()  const {        return P_left_vehicle_;    }
    inline PoseD P_right_vehicle() const {
        return P_left_vehicle_.q().append(P_left_vehicle_.t() + Vector3d(-baseline_,0,0));
    }
    inline PoseD P_right_left()    const {        return PoseD(Vector3d(-baseline_,0,0));    }





    StereoCalibration()=default;
    StereoCalibration(int rows_, int cols_,
                      double fy_, double fx_, double py_, double px_,
                      double baseline_,
                      PoseD P_left_vehicle_):
        rows_(rows_), cols_(cols_),
        fy_(fy_), fx_(fx_),py_(py_), px_(px_),
        baseline_(baseline_),
        P_left_vehicle_(P_left_vehicle_){}

    int rows()const {return rows_;}
    int cols()const {return cols_;}
    // these should be fully exposed to ensure
    // compiler visibility for optimization.
    inline double fy() const {return fy_;}
    inline double fx() const {return fx_;}
    inline double py() const {return py_;}
    inline double px() const {return px_;}
    // in meters!
    inline double baseline()const {return baseline_;}
    template<class T> inline T disparity(T depth) const{        return (fx_*baseline_)/depth;    }


    // from normalized coordinates to row,col
    template<class T> inline Vector2<T> distort(Vector2<T> yn)  const{

        T row=T(fy())*yn[1] + T(py());
        T col=T(fx())*yn[0] + T(px());
        return {row,col};
    }
    // from row, col to normalized coordinates
    template<class T> inline Vector2<T> undistort(Vector2<T> y) const {
        // row, col,
        T c1=(y[0] - T(py()))/T(fy());
        T c0=(y[1] - T(px()))/T(fx());
        return {c0,c1};  // yn
    }
    template<class T> inline Vector2<T> undistort(T row, T col) const {
        // row, col,
        T c1=(row - T(py()))/T(fy());
        T c0=(col - T(px()))/T(fx());
        return {c0,c1};  // yn
    }


    // from Xcam 3d point to row,col, I use separate ones, because the inlining limit can make trouble otherwise.
    template<class T>  inline Vector2<T> project_cam(Vector3<T> x_cam) const {
        T z=x_cam[2];
        T row=T(fy_)*x_cam[1]/z + T(py_);
        T col=T(fx_)*x_cam[0]/z + T(px_);
        return {row,col};
    }
    template<class T>  inline Vector2<T> project_cam(Vector4<T> x_cam) const {
        // same as project, the last term isnt used.
        //This is because dividing with it is just always the wrong thing to do both geometrically and computationally
        T z=x_cam[2];
        T row=T(fy_)*x_cam[1]/z + T(py_);
        T col=T(fx_)*x_cam[0]/z + T(px_);
        return {row,col};
    }
    template<class T>  inline T disparity_cam(Vector3<T> x_cam) const  {        return disparity(x_cam[2]);    }
    template<class T>  inline T disparity_cam(Vector4<T> x_cam) const  {        return disparity(x_cam[2]/x_cam[3]);    }
    template<class T>  inline Vector2<T> project_right_cam(Vector3<T> x_cam) const  {
        x_cam[0]-=T(baseline_);
        return project_cam(x_cam);
    }
    template<class T>  inline Vector2<T> project_right_cam(Vector4<T> x_cam) const  {
        x_cam[0]-=(T(baseline_)*x_cam[3]);
        return project_cam(x_cam);
    }
    template<class T>  inline Vector3<T> stereo_project_cam(Vector3<T> x_cam) const {
        Vector2<T> l=project_cam(x_cam);
        return Vector3<T>(l[0],l[1],disparity_cam(x_cam)); // disparity in col
    }
    template<class T>  inline Vector3<T> stereo_project_cam(Vector4<T> x_cam) const {
        Vector2<T> l=project_cam(x_cam);
        return Vector3<T>(l[0],l[1],disparity_cam(x_cam)); // disparity in col
    }

    //////////// from vehicle to camera,
    template<class T> inline Vector3<T> x_cam_vehicle(Vector3<T> Xv) const{
        return Pose<T>(P_left_vehicle_)*Xv;
    }
    template<class T> inline Vector4<T> x_cam_vehicle(Vector4<T> Xv) const{
        return Pose<T>(P_left_vehicle_)*Xv;
    }
    template<class T> inline Vector3<T> x_right_vehicle(Vector3<T> Xv) const{
        auto x=x_cam_vehicle(Xv);
        x[0]-=T(baseline_);
        return x;
    }

    template<class T> inline Vector4<T> x_right__vehicle(Vector4<T> Xv) const{
        auto x=x_cam_vehicle(Xv);
        x[0]-=T(baseline_)*x[3];
        return x;
    }

    template<class T>  inline Vector2<T> project(Vector3<T> Xv) const {
        return project_cam(x_cam_vehicle(Xv));
    }
    template<class T>  inline Vector2<T> project(Vector4<T> Xv)    const     {
        return project_cam(x_cam_vehicle(Xv));
    }
    template<class T>  inline Vector2<T> project_right(Vector3<T> Xv) const  {
        return project_cam(x_right_vehicle(Xv));
    }
    template<class T>  inline Vector2<T> project_right(Vector4<T> Xv) const  {
        return project_cam(x_right_vehicle(Xv));
    }
    template<class T>  inline T disparity(Vector3<T> Xv) const  {
        return disparity_cam(x_cam_vehicle(Xv));
    }
    template<class T>  inline T disparity(Vector4<T> Xv) const  {
        return disparity_cam(x_cam_vehicle(Xv));
    }
    template<class T>  inline Vector3<T> stereo_project(Vector3<T> Xv) const {
        return stereo_project_cam(x_cam_vehicle(Xv));
    }
    template<class T>  inline Vector3<T> stereo_project(Vector4<T> Xv) const {
        return stereo_project_cam(x_cam_vehicle(Xv));
    }

    ////////////////behind camera
    template<class T> inline bool behind_cam (Vector3<T> Xc) const{        return Xc[2]<T(0);    }
    template<class T> inline bool behind_cam (Vector4<T> Xc) const{        return Xc[2]<Xc[3];   }

    template<class T> inline bool behind (Vector3<T> Xv) const{      return   behind_cam(x_cam_vehicle(Xv));    }
    template<class T> inline bool behind (Vector4<T> Xv) const{      return   behind_cam(x_cam_vehicle(Xv));    }

    // the old behind either, was incorrect!


    //////////////////// Triangulation, these names were bad

    inline Vector4d  triangulate_x_camera(double row, double col, double disparity) const {
        return ::cvl::triangulate_ray(undistort(row, col),fx(),baseline(),disparity).normalized();
    }
    inline Vector4d  triangulate_x_camera(Vector2d y /*row, col*/, double disparity) const {
        return ::cvl::triangulate_ray(undistort(y[0],y[1]),fx(),baseline(),disparity).normalized();
    }
    inline Vector4d  triangulate_x_camera(Vector3d y/*row, col, disp*/) const {
        return ::cvl::triangulate_ray(undistort(y[0],y[1]),fx(),baseline(),y[2]).normalized();
    }

    inline Vector4d  triangulate_x_vehicle(double row, double col, double disparity) const {
        return P_left_vehicle_.inverse()*triangulate_x_camera(row,col,disparity);
    }
    inline Vector4d  triangulate_x_vehicle(Vector2d y /*row, col*/, double disparity) const {
        return P_left_vehicle_.inverse()*triangulate_x_camera(y,disparity);
    }
    inline Vector4d  triangulate_x_vehicle(Vector3d y/*row, col, disp*/) const {
        return P_left_vehicle_.inverse()*triangulate_x_camera(y);
    }


    //////////////// for ceres, reprojection costs, accounts for behind too...




    std::string str() {
        std::stringstream ss;
        ss<<"Stereo Calibration: \n";
        ss<<"rows: =    "<<rows_<<"\n";
        ss<<"cols: =    "<<cols_<<"\n";
        ss<<"f_row =    "<<fy_<<"\n";
        ss<<"fx =    "<<fx_<<"\n";
        ss<<"p_row =    "<<py_<<"\n";
        ss<<"px =    "<<px_<<"\n";
        ss<<"baseline=  "<<baseline_<<"\n";
        ss<<"P_cam0_imu="<<P_left_vehicle_;
        return ss.str();
    }
};


} // end namespace cvl

