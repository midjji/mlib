#pragma once
#include <mlib/utils/cvl/matrix.h>
#include <mlib/utils/cvl/pose.h>
#include <ceres/ceres.h>
#include <ceres/cost_function.h>
namespace cvl {





template<class Intrinsics, int StateSize>
class ReprojectionCost
{
public:
    Intrinsics intrinsics;
    Vector2d y; // (row,col,(col-disparity))
    static constexpr int resids=2;
    ReprojectionCost()=default;
    ReprojectionCost(Intrinsics intrinsics, Vector2d y):intrinsics(intrinsics),y(y){}
    template <typename T>
    bool operator()(const T* const Pvw_, // just pose works thanks to ceres improvment! // but is quite possibly slower...
                    const T* const Xw_,
                    T* residuals) const
    {
        Pose<T> Pvw(Pvw_,true);
        Vector<T,StateSize> Xc=intrinsics.x_cam_vehicle(Pvw*Vector<T,StateSize>::copy_from(Xw_));

        // the critically important things are
        // 1 behind should be slightly penalized
        // 2 the poses should be less penalized to avoid outliers causing bad poses
        // 3 are the gradients in the right directions for the negative case? I have no idea
        if(intrinsics.behind_cam(Xc))
        {
            residuals[0]=T(0);
            residuals[1]=T(0.1)*(intrinsics.disparity_cam(Xc) - T(1));
            return true;
        }

        Vector2<T> yp=intrinsics.project_cam(Xc);
        for(int i=0;i<resids;++i)
            residuals[i]=T(y[i])  - yp[i];
        return true;
    }
};
template<class Intrinsics,
         int StateSize>
class StereoCost
{
public:
    Intrinsics intrinsics;
    Vector3d y; // (row,col,(col-disparity)), note disparity is assumed positive, and this is guaranteed if the reprojection_cost creator is used!
    static constexpr int resids=3;
    StereoCost()=default;
    StereoCost(Intrinsics intrinsics, Vector3d y):intrinsics(intrinsics),y(y){}
    template <typename T>
    bool operator()(const T* const Pvw_, // just pose works thanks to ceres improvment! // but is quite possibly slower...
                    const T* const Xw_,
                    T* residuals) const
    {
        Pose<T> Pvw(Pvw_,true);
        Vector<T,StateSize> Xc=intrinsics.x_cam_vehicle(Pvw*Vector<T,StateSize>::copy_from(Xw_));

        // the critically important things are
        // 1 behind should be slightly penalized
        // 2 the poses should be less penalized to avoid outliers causing bad poses
        // 3 are the gradients in the right directions for the negative case? I have no idea
        if(intrinsics.behind_cam(Xc))
        {
            residuals[0]=T(0);
            residuals[1]=T(0);
            residuals[2]=T(0.1)*(intrinsics.disparity_cam(Xc) - T(y[2]));
            return true;
        }

        Vector3<T> yp=intrinsics.stereo_project_cam(Xc);
        for(int i=0;i<3;++i)
            residuals[i]=T(y[i])  - yp[i];
        return true;
    }
};




template<class Intrinsics, int StateSize=4> auto reprojection_cost(Intrinsics intrinsics, Vector2d y){
    using cost=ReprojectionCost<Intrinsics,StateSize>;
    return (new ceres::AutoDiffCostFunction<cost, cost::resids, 7, StateSize >(
                new cost(intrinsics, y)));
}
template<class Intrinsics, int StateSize=4> ceres::CostFunction* reprojection_cost(Intrinsics intrinsics, Vector3d y)
{
    if(y[2]<0) return reprojection_cost(intrinsics, y.drop_last());

    using cost=StereoCost<Intrinsics,StateSize>;
    // resid,first param,second param, third param
    return (new ceres::AutoDiffCostFunction<cost, cost::resids, 7, StateSize >(
                new cost(intrinsics, y)));
}

template<class Intrinsics>
class PoseCost
{
public:
    Intrinsics intrinsics;
    Vector2d y; // (row,col,(col-disparity))
    Vector4d xw;
    static constexpr int resids=2;
    PoseCost()=default;
    PoseCost(Intrinsics intrinsics, Vector2d y, Vector4d xw):intrinsics(intrinsics),y(y),xw(xw){}
    template <typename T>
    bool operator()(const T* const pose,
                    T* residuals) const
    {
        Pose<T> Pvw(pose,true);
        Vector4<T> Xw(xw);
        Vector2<T> yp=intrinsics.project(Pvw*Xw);
        for(int i=0;i<2;++i)
            residuals[i]=T(y[i])  - yp[i];
        return true;
    }
};
template<class Intrinsics>
class StereoPoseCost
{
public:
    Intrinsics intrinsics;
    Vector3d y; // (row,col,(col-disparity))
    Vector4d xw;
    static constexpr int resids=3;
    StereoPoseCost()=default;
    StereoPoseCost(Intrinsics intrinsics, Vector3d y, Vector4d xw):intrinsics(intrinsics),y(y),xw(xw){}
    template <typename T>
    bool operator()(const T* const pose,
                    T* residuals) const
    {
        Pose<T> Pvw(pose,true);
        Vector4<T> Xw(xw);
        Vector3<T> yp=intrinsics.stereo_project(Pvw*Xw);
        for(int i=0;i<resids;++i)
            residuals[i]=T(y[i])  - yp[i];
        return true;
    }
};

template<class Intrinsics, int StateSize=4> auto reprojection_cost(Intrinsics intrinsics, Vector2d y,Vector4d xw){
    using cost=PoseCost<Intrinsics>;
    return (new ceres::AutoDiffCostFunction<cost, cost::resids, 7>(
                new cost(intrinsics, y,xw)));
}

template<class Intrinsics, int StateSize=4> ceres::CostFunction* reprojection_cost(Intrinsics intrinsics, Vector3d y,Vector4d xw)
{
    if(y[2]<0) return reprojection_cost(intrinsics, y.drop_last(),xw);

    using cost=StereoPoseCost<Intrinsics>;
    // resid,first param,second param, third param
    return (new ceres::AutoDiffCostFunction<cost, cost::resids, 7>(
                new cost(intrinsics, y,xw)));
}

template<class Intrinsics, int StateSize=4> auto reprojection_cost(Intrinsics intrinsics, Vector2d y, Vector3d xw){
    return reprojection_cost(intrinsics, y, xw.homogeneous().normalized());
}

template<class Intrinsics, int StateSize=4> ceres::CostFunction* reprojection_cost(Intrinsics intrinsics, Vector3d y,Vector3d xw)
{
    return reprojection_cost(intrinsics, y, xw.homogeneous().normalized());
}

template<class Intrinsics, int StateSize>
class TriangulationCost
{
public:
    PoseD Pvw;
    Vector3d y; // (row,col,disparity)
    Intrinsics intrinsics;
    static constexpr int resids=2;


    TriangulationCost(Intrinsics intrinsics, PoseD Pvw, Vector3d y):Pvw(Pvw),y(y),intrinsics(intrinsics){}
    template <typename T>
    bool operator()(const T* const Xw,
                    T* residuals) const
    {
        Pose<T> Pvw_(Pvw);
        Vector2<T> yp=intrinsics.project(Pvw_*(Vector<T,4>::copy_from(Xw)));
        for(int i=0;i<resids;++i)
            residuals[i]=T(y[i]) - yp[i];
        return true;
    }
};



template<class Intrinsics, int StateSize>
class StereoTriangulationCost
{
public:
    PoseD Pvw;
    Vector3d y; // (row,col,disparity)
    Intrinsics intrinsics;
    static constexpr int resids=3;


    StereoTriangulationCost(Intrinsics intrinsics, PoseD Pvw, Vector3d y):Pvw(Pvw),y(y),intrinsics(intrinsics){}
    template <typename T>
    bool operator()(const T* const Xw,
                    T* residuals) const
    {
        Pose<T> Pvw_(Pvw);
        Vector3<T> yp=intrinsics.stereo_project(Pvw_*(Vector<T,4>::copy_from(Xw)));
        for(int i=0;i<resids;++i)
            residuals[i]=T(y[i]) - yp[i];
        return true;
    }

};

template<class Intrinsics, int StateSize=4> ceres::CostFunction* reprojection_cost(Intrinsics intrinsics, PoseD Pvw, Vector2d y){
    using cost=TriangulationCost<Intrinsics,StateSize>;
    return (new ceres::AutoDiffCostFunction<cost, cost::resids, StateSize >(
                new cost(intrinsics, Pvw, y)));
}

template<class Intrinsics, int StateSize=4> ceres::CostFunction* reprojection_cost(Intrinsics intrinsics, PoseD Pvw, Vector3d y){
    if(y[2]<0) return reprojection_cost(intrinsics, y.drop_last());
    using cost=StereoTriangulationCost<Intrinsics,StateSize>;
    return (new ceres::AutoDiffCostFunction<cost, cost::resids, StateSize >(
                new cost(intrinsics, Pvw, y)));
}




}
