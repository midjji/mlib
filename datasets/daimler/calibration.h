#pragma once

/* ********************************* FILE ************************************/
/** \file    calibration.h
 *
 * \brief    This header contains the calibration for the sequences I got from daimler after postprocessing...
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
 * @brief The DaimlerCalibration class
 * Can be made faster using constexpr, but it makes it almost impossible to use...
 * assumes row, col
 */
class DaimlerCalibration{

    /*
    // the row, col one
    Matrix3d K{0, 2261.54, 1108.15,
               2267.22, 0, 519.169,
               0, 0, 1}; // for both cameras
    */

    double fy_=2261.54;
    double fx_=2267.22;

    double py_=1108.15;
    double px_=519.169;
    double baseline_=0.209114;

public:
    double fy() const {return fy_;}
    double fx() const {return fx_;}
    double py() const {return py_;}
    double px() const {return px_;}
    // in meters!
    double baseline()const {return baseline_;}

    // from 3d point to row,col
    template<class T>  Vector2<T> project(Vector3<T> x) const {
        // not multiplying by zeros matters for optimization
        T row=T(fy())*x[1]/x[2] + T(py());
        T col=T(fx())*x[0]/x[2] + T(px());
        return {row,col};
        //Vector2<T> b=(Matrix3<T>(K)*x).dehom();
    }
    // from normalized coordinates to row,col
    template<class T> Vector2<T> distort(Vector2<T> yn) const{
        T row=T(fy())*yn[1] + T(py());
        T col=T(fx())*yn[0] + T(px());
        return {row,col};
    }

    template<class T>  Vector2<T> undistort(Vector2<T> y) const {
        // row, col,
        T c1=(y[0] - T(py()))/T(fy());
        T c0=(y[1] - T(px()))/T(fx());
        return {c0,c1};  // yn

        //Vector2<T> xx=(Matrix3<T>(Kinv)*y.homogeneous()).dehom();
    }


    template<class T>  Vector3<T> stereo_project(Vector3<T> x) const {
        Vector2<T> l=project(x);
        Vector2<T> r=project_right(x);
        return Vector3<T>(l[0],l[1],(l[1]-r[1])); // disparity in col
    }
    template<class T>  Vector3<T> stereo_project(Vector4<T> x) const {

        Vector2<T> l=project(x);
        Vector2<T> r=project_right(x);
        return Vector3<T>(l[0],l[1],(l[1]-r[1])); // disparity in col
    }

    template<class T>  Vector2<T> project_right(Vector3<T> x)const  {
        x[0]-=T(baseline());
        return project(x);
    }
    template<class T>  Vector2<T> project_right(Vector4<T> x)const  {

        x[0]-=(T(baseline())*x[3]);        
        return project(x);
    }
    template<class T>  Vector2<T> project(Vector4<T> x)    const     {

        // its a ray!
        //return project(x.drop_last());
        return project(x.dehom());
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
    Vector4d  triangulate_ray_from_yn(Vector2d yn, double disparity)const {
        return ::cvl::triangulate_ray(yn,fx(),baseline(),disparity);
    }

    bool global_shutter() const {
        return true;
    }
    unsigned int rows()const {return 1024;}
    unsigned int cols()const {return 2048;}


    PoseD P_cam0_vehicle()const {
        // roughly right... not perfect...
        return PoseD(getRotationMatrixY(13.0 * 3.1415/180.0)*getRotationMatrixX(15.0 * 3.1415/180.0));
    }
    static DaimlerCalibration common_cal;
    static const DaimlerCalibration& common(){return common_cal;}

};


} // end namespace cvl
