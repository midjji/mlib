#pragma once

#include <mlib/utils/cvl/pose.h>
#include <mlib/utils/cvl/triangulate.h>
#include <mlib/datasets/stereo_calibration.h>

namespace cvl{
namespace hilti {



class Calibration
{
public:
    int rows_=1080;
    int cols_=1440;




    double fy_=527.8697573702394;
    double fx_=527.8697573702394;
    double py_=546.6633124512576;
    double px_=725.2570598632401;
    double baseline_= 1.06837645e-01;
    PoseD P_left_imu_;
    PoseD P_right_imu_;
    PoseD P_cam2_imu_;
    PoseD P_cam3_imu_;
    PoseD P_cam4_imu_;

    PoseD& P_x_imu(int i)
    {
        switch (i){
        case 0: return P_left_imu_;
        case 1: return P_right_imu_;
        case 2: return P_cam2_imu_;
        case 3: return P_cam3_imu_;
        case 4: return P_cam4_imu_;
        }
    }

    StereoCalibration stereo_calibration(int index) const{
        return StereoCalibration(rows_, cols_, fy_,fx_,py_,px_,baseline_, P_x_imu(index));
    }

    inline const PoseD& P_x_imu(int i) const
    {
        switch (i){
        case 0: return P_left_imu_;
        case 1: return P_right_imu_;
        case 2: return P_cam2_imu_;
        case 3: return P_cam3_imu_;
        case 4: return P_cam4_imu_;
        default: wtf();return P_left_imu_;
        }
    }


    Calibration()=default;
    Calibration(int rows_,int cols_,
                double fy_,
                double fx_,
                double py_,
                double px_,
                double baseline_):
        rows_(rows_),cols_(cols_),fy_(fy_),fx_(fx_),py_(py_),
        px_(px_),baseline_(baseline_){}

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
        return project_cam(Pose<T>(P_left_imu_)*x);
    }
    template<class T>  inline Vector2<T> project(Vector4<T> x)    const     {
        // its a ray!
        return project_cam((Pose<T>(P_left_imu_))*x);
    }


    template<class T>  inline Vector2<T> project_right(Vector3<T> x) const  {
        return project_cam(Pose<T>(P_left_imu_)*x);
    }
    template<class T>  inline Vector2<T> project_right(Vector4<T> x) const  {
        return project_cam(Pose<T>(P_left_imu_)*x);
    }
    template<class T> inline Vector3<T> x_cam_vehicle(Vector3<T> x) const{
        return P_left_imu_*x;
    }
    template<class T> inline Vector4<T> x_cam_vehicle(Vector4<T> x) const{
        return P_left_imu_*x;
    }
    template<class T>  inline T disparity(Vector3<T> x) const  {
        return disparity_cam(x_cam_vehicle(x));
    }
    template<class T>  inline T disparity(Vector4<T> x) const  {
        return disparity_cam(x_cam_vehicle(x));
    }


    template<class T>  inline Vector3<T> stereo_project(Vector3<T> x) const {
        x=Pose<T>(P_left_imu_)*x;
        Vector2<T> l=project_cam(x);
        Vector2<T> r=project_right_cam(x);
        return Vector3<T>(l[0],l[1],l[1]-r[1]); // disparity in col
    }
    template<class T>  inline Vector3<T> stereo_project(Vector4<T> x) const {
        x=Pose<T>(P_left_imu_)*x;
        Vector2<T> l=project_cam(x);
        Vector2<T> r=project_right_cam(x);
        return Vector3<T>(l[0],l[1],l[1]-r[1]); // disparity in col
    }

    template<class T>
    inline bool behind_either( Vector3<T> x) const{
        x=P_left_imu_*x;
        return x[2]<baseline();
    }
    template<class T>
    inline bool behind_either(Vector4<T> x) const{
        x=P_left_imu_*x;
        return x[2]<baseline()*x[3];
    }


    bool behind_either(Vector3d x) const{
        return (x[2]<baseline());
    }


    inline Vector4d  triangulate_ray(Vector2d rowcol, double disparity)const {
        return ::cvl::triangulate_ray(undistort(rowcol),fx(),baseline(),disparity);
    }

    inline Vector4d  triangulate_ray(Vector3d rowcoldisp)const {
        return triangulate_ray(Vector2d(rowcoldisp[0],rowcoldisp[1]),rowcoldisp[2]);
    }

    inline Vector4d  triangulate_ray_from_yn(Vector2d yn, double disparity) const {
        //::cvl::triangulate_ray(yn,fx(),baseline(),disparity);
        const Vector4d x_cam(    yn[0],
                yn[1],
                (1.0),
                disparity/(fx_*baseline_));
        return P_left_imu_.inverse()*x_cam;
    }
    std::string str() {
        std::stringstream ss;ss.precision(19);
        ss<<"Hilti Calibration: \n";
        ss<<"rows: =    "<<rows_<<"\n";
        ss<<"cols: =    "<<cols_<<"\n";
        ss<<"f_row =    "<<fy_<<"\n";
        ss<<"f_col =    "<<fx_<<"\n";
        ss<<"p_row =    "<<py_<<"\n";
        ss<<"p_col =    "<<px_<<"\n";
        ss<<"baseline=  "<<baseline_<<"\n";
        ss<<"P_left_imu_="<<P_left_imu_<<"\n";
        ss<<"P_righ_imu="<<P_right_imu_<<"\n";
        ss<<"P_cam2_imu="<<P_cam2_imu_<<"\n";
        ss<<"P_cam3_imu="<<P_cam3_imu_<<"\n";
        ss<<"P_cam4_imu="<<P_cam4_imu_<<"\n";
        return ss.str();
    }
};

}
} // end namespace cvl
