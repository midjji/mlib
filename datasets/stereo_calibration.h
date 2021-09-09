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
#include <mlib/utils/cvl/pose.h>
#include <mlib/utils/cvl/triangulate.h>

namespace cvl{
/**
 * @brief The StereoCalibration class
 * Assumes pinholenormalized images...
 */
class StereoCalibration
{

    /*
    // the row, col one
    Matrix3d K{0, 2261.54, 1108.15,
               2267.22, 0, 519.169,
               0, 0, 1}; // for both cameras
    */

    int rows_=1024;
    int cols_=2048;

    double fy_=2261.54;
    double fx_=2267.22;
    double py_=1108.15;
    double px_=519.169;
    double baseline_=0.209114;
    PoseD P_cam0_vehicle_;

public:
    StereoCalibration()=default;
    StereoCalibration(int rows_,int cols_,
                      double fy_,
                      double fx_,
                      double py_,
                      double px_,
                      double baseline_,
                      PoseD P_cam0_vehicle_):
        rows_(rows_),cols_(cols_),fy_(fy_),fx_(fx_),py_(py_),
        px_(px_),baseline_(baseline_),
        P_cam0_vehicle_(P_cam0_vehicle_){}

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
    double disparity(double depth) const{ return depth/(fx()*baseline());}


    // from normalized coordinates to row,col
    template<class T>  inline Vector2<T> distort(Vector2<T> yn) const{
        T row=T(fy())*yn[1] + T(py());
        T col=T(fx())*yn[0] + T(px());
        return {row,col};
    }

    template<class T>  inline Vector2<T> undistort(Vector2<T> y) const {
        // row, col,
        T c1=(y[0] - T(py()))/T(fy());
        T c0=(y[1] - T(px()))/T(fx());
        return {c0,c1};  // yn
        //Vector2<T> xx=(Matrix3<T>(Kinv)*y.homogeneous()).dehom();
    }
    template<class T> inline Vector2<T> undistort(T row, T col) const {
        // row, col,
        T c1=(row - T(py()))/T(fy());
        T c0=(col - T(px()))/T(fx());
        return {c0,c1};  // yn
    }

    // from 3d point to row,col
    template<class T>  inline Vector2<T> project_cam(Vector3<T> x) const {
        // not multiplying by zeros matters for optimization
        T z=x[2];
        //if(z<T(0)) z=-z;
        T row=T(fy())*x[1]/z + T(py());
        T col=T(fx())*x[0]/z + T(px());
        return {row,col};
        //Vector2<T> b=(Matrix3<T>(K)*x).dehom();
    }
    template<class T>  inline Vector2<T> project_cam(Vector4<T> x) const {
        // same as project, the last term isnt used.
        T z=x[2];
        T row=T(fy())*x[1]/z + T(py());
        T col=T(fx())*x[0]/z + T(px());
        return {row,col};
    }
    template<class T>  inline Vector2<T> project_right_cam(Vector3<T> x) const  {
        x[0]-=T(baseline());
        return project_cam(x);
    }
    template<class T>  inline Vector2<T> project_right_cam(Vector4<T> x) const  {
        x[0]-=(T(baseline())*x[3]);
        return project_cam(x);
    }
    template<class T>  inline T disparity_cam(Vector3<T> x) const  {
        return T(fx()*baseline())/x[2];
    }
    template<class T>  inline T disparity_cam(Vector4<T> x) const  {
        return T(fx()*baseline())/x[2];
    }

    template<class T>  inline Vector2<T> project(Vector3<T> x) const {
        return project_cam(Pose<T>(P_cam0_vehicle_)*x);
    }
    template<class T>  inline Vector2<T> project(Vector4<T> x)    const     {
        // its a ray!
        return project_cam((Pose<T>(P_cam0_vehicle_))*x);
    }


    template<class T>  inline Vector2<T> project_right(Vector3<T> x) const  {
        return project_cam(Pose<T>(P_cam0_vehicle_)*x);
    }
    template<class T>  inline Vector2<T> project_right(Vector4<T> x) const  {
        return project_cam(Pose<T>(P_cam0_vehicle_)*x);
    }
    template<class T> inline Vector3<T> x_cam_vehicle(Vector3<T> x) const{
        return P_cam0_vehicle_*x;
    }
    template<class T> inline Vector4<T> x_cam_vehicle(Vector4<T> x) const{
        return P_cam0_vehicle_*x;
    }
    template<class T>  inline T disparity(Vector3<T> x) const  {
        return disparity_cam(x_cam_vehicle(x));
    }
    template<class T>  inline T disparity(Vector4<T> x) const  {
        return disparity_cam(x_cam_vehicle(x));
    }


    template<class T>  inline Vector3<T> stereo_project(Vector3<T> x) const {
        x=Pose<T>(P_cam0_vehicle_)*x;
        Vector2<T> l=project_cam(x);
        Vector2<T> r=project_right_cam(x);
        return Vector3<T>(l[0],l[1],l[1]-r[1]); // disparity in col
    }
    template<class T>  inline Vector3<T> stereo_project(Vector4<T> x) const {
        x=Pose<T>(P_cam0_vehicle_)*x;
        Vector2<T> l=project_cam(x);
        Vector2<T> r=project_right_cam(x);
        return Vector3<T>(l[0],l[1],l[1]-r[1]); // disparity in col
    }

    template<class T>
    inline bool behind_either( Vector3<T> x) const{
        x=P_cam0_vehicle_*x;
        return x[2]<baseline();
    }
    template<class T>
    inline bool behind_either(Vector4<T> x) const{
        x=P_cam0_vehicle_*x;
        return x[2]<baseline()*x[3];
    }


    bool behind_either(Vector3d x) const{
        return (x[2]<baseline());
    }


    Vector4d  triangulate_ray(Vector2d rowcol, double disparity)const {
        return ::cvl::triangulate_ray(undistort(rowcol),fx(),baseline(),disparity);
    }
    Vector4d  triangulate_ray(Vector3d rowcoldisp)const {
        return triangulate_ray(Vector2d(rowcoldisp[0],rowcoldisp[1]),rowcoldisp[2]);
    }
    inline Vector4d  triangulate_ray_from_yn(Vector2d yn, double disparity) const {
                //::cvl::triangulate_ray(yn,fx(),baseline(),disparity);
        const Vector4d x_cam(    yn[0],
                    yn[1],
                    (1.0),
                    disparity/(fx_*baseline_));
        return P_cam0_vehicle_.inverse()*x_cam;
    }

    PoseD P_cam0_vehicle()const {
        return P_cam0_vehicle_;

    }
};


} // end namespace cvl
