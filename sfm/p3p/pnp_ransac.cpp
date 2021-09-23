#include <iostream>
#include <ceres/ceres.h>

#include <mlib/utils/mlibtime.h>
#include <mlib/utils/random.h>
#include <mlib/sfm/p3p/pnp_ransac.h>
#include <mlib/sfm/p3p/p4p.h>

#include <thread>


using std::cout;using std::endl;


namespace cvl {

PoseD pnp_ransac(const std::vector<cvl::Vector3d>& xs,
                 const std::vector<cvl::Vector2d>& yns,
                 PnpParams params){
    if(xs.size()<4) return PoseD::Identity();
    PNP pnp(params);
    return pnp.compute(xs,yns);
}

PoseD pnp_ransac(const std::vector<cvl::Vector4d>& xs,
                 const std::vector<cvl::Vector2d>& yns,
                 PnpParams params){
    std::vector<cvl::Vector3d> txs;txs.reserve(xs.size());
    std::vector<cvl::Vector2d> tys;tys.reserve(yns.size());
    for(uint i=0;i<xs.size();++i)
    {
        assert(xs.size()==yns.size());

        auto x3=xs[i].dehom();

        if(x3.isnormal())
        {
            txs.push_back(x3);
            tys.push_back(yns[i]);
        }
    }
    if(txs.size()<xs.size()) cout<<"lost due to ray!"<<xs.size()<< " "<<txs.size() <<endl;;
    return pnp_ransac(txs,tys, params);
}


/**
 * @brief evaluate_inlier_set
 * @param xs
 * @param yns
 * @param threshold
 * @param pose
 * @param best_inliers
 * @return number of inliers if greater than best_inliers, otherwize slightly less...
 * set best_inliers to 0 to compute the exact count...
 *
 */
int evaluate_inlier_set(const std::vector<cvl::Vector3d>& xs,
                        const std::vector<cvl::Vector2d>& yns,
                        double threshold,
                        PoseD pose,
                        uint best_inliers){
    // this is the by far slowest part of the system!



    uint inliers=0;


    Matrix4d M=pose.get4x4(); // much faster...


    double threshold_squared=threshold*threshold; // we are comparing to the square of it after all...

    for(uint i=0;i<xs.size();++i){

        //cout<<((pose*xs[i]).dehom() - yns[i]).squaredNorm()<<"   "<<threshold_squared<<endl;

        cvl::Vector4d X=xs[i].homogeneous(); // yes even with the extra cost here...
        //        Vector3d XR=(M*X).dehom(); // technically based on how 4x4 etc work, no dehom required
        Vector4d XR=(M*X);





        double x=XR[0];
        double y=XR[1];
        // any negative value is behind the camera, those are outliers by any definition!
        // however very distant points behave fairly well... even with the abs... so break is needed
        double iz=1.0/XR[2];

        if(iz<0) continue;


        double err1=x *iz - yns[i](0);
        double err2=y *iz - yns[i](1);

        double err=err1*err1 + err2*err2;

        inliers += (err < threshold_squared) ? 1 : 0;
        //mle += std::min(errors[i],thr);// use this to compute mle instead...
        // highest number of inliers possible at this point. inliers + (xs.size()) -i
        //if(((xs.size()-i +inliers)<best_inliers)) break;
    }
    return inliers;

}

/**
 * @brief The PnPReprojectionError class pnp cost for ceres
 *
 * Note:
 * - The cost is will become vectorized if compiled with optimization
 *
 */
class PnPReprojectionError
{


public:
    /**
     * @brief PnPReprojectionError
     * @param xs
     * @param yns
     */
    PnPReprojectionError(const std::vector<Vector3d>& xs,
                         const std::vector<Vector2d>& yns):xs(xs),yns(yns)  {}

    template <typename T>
    /**
     * @brief operator () autodiff enabled error
     * @param rotation
     * @param translation
     * @param residuals
     * @return
     */
    bool operator()(const T* const rotation, const T* const translation,T* residuals) const
    {
        // Get camera rotation and translation
        cvl::Pose<T> P(rotation,translation);
        //cvl::Matrix3<T> R=P.getR();
        cvl::Matrix4<T> M=P.get4x4();
        //cvl::Vector3<T> tr(translation,true);
        for (uint i = 0; i < xs.size(); ++i) {

            cvl::Vector4<T> x=xs[i].homogeneous();

            cvl::Vector4<T> xr=M*x;
            T iz=T(1.0)/xr[2];

            if(iz<T(0))
            {
                // give it a chance to recover at low impact
                residuals[0]=T(1e-6)*(T(1)-iz);
                residuals[1]=T(0);
                return true;
            }
            residuals[0] = xr[0] *iz - T(yns[i][0]);
            residuals[1] = xr[1] *iz - T(yns[i][1]);
            residuals+=2;
        }
        return true;
    }
    /// the 3d point observations
    std::vector<Vector3d> xs;
    /// the pinhole normalized image observations
    std::vector<Vector2d> yns;
    /**
     * @brief Create Autodiff error factory
     * @param inlier_xs
     * @param inlier_yns
     * @return
     */
    static ceres::CostFunction* Create(const std::vector<Vector3d>& inlier_xs,
                                       const std::vector<Vector2d>& inlier_yns ){
        return new ceres::AutoDiffCostFunction<PnPReprojectionError, ceres::DYNAMIC, 4,3>(
                    new PnPReprojectionError(inlier_xs,inlier_yns), inlier_xs.size()*2);
    }

};




Vector4<uint> get4RandomInRange0(uint max){

    Vector4<uint> indexes;
    //todo verify the fast one!
    /*
    n=0;
    while(n<3){
        uint val=randui<int>(0,max-1);
        for(int i=0;i<n;++i)
            if(indexes[i]==val) continue;
        indexes[n++]=val;
    }*/

    // for large numbers in smallish sets, sorting is faster
    std::set<uint> set;
    assert(4<=max);
    if(max<4)
        std::cout<<"called pnp with less than 4 points, something is wrong!"<<std::endl;
    while(set.size()<4)
        set.insert(mlib::randui(0,max-1));
    int n=0;
    for(uint i:set){
        indexes[n++]=i;
    }
    return indexes;
}


PNP::PNP(PnpParams params ):params(params){}
PoseD PNP::operator()(const std::vector<Vector3d>& xs,/// the known 3d positions in world coordinates
const std::vector<Vector2d>& yns/// the pinhole normalized measurements corresp to xs
){return compute(xs,yns);}

PoseD PNP::compute(const std::vector<Vector3d>& xs,/// the known 3d positions in world coordinates
                   const std::vector<Vector2d>& yns/// the pinhole normalized measurements corresp to xs
                   ){
    /// number of inliers of best solution
    double best_inliers=0;
    /// the best pose so far
    PoseD best_pose=PoseD::Identity();
    if(xs.size()<4) return best_pose;
    if(yns.size()!=xs.size()){
        mlog()<<"number of x,yn pairs must match\n";
        exit(1);
    }

    int total_iters=0;
    params.update_all(); // read any new parameters, in case the user has changed something via e.g. gui...

    double threshold=params.threshold->value();

    int max_iters=params.get_iterations(0.1);

    if(max_iters>1000)
    {
        max_iters=1000;
    }

    int i;

    //mlib::Timer timer("iteration time: ");

    for(i=0;i<max_iters;++i)
    {
        //timer.tic();
        // pick 4 at random,

        // will always succeed, returns identity on degeneracy...
        PoseD pose=p4p(xs,yns,get4RandomInRange0(xs.size()), params.max_angle, params.reference);


        assert(pose.is_normal());
        if(!pose.is_normal()) continue;
        //cout<<pose<<endl;

        // evaluate inlier set
        // timer.tic();

        uint inliers=evaluate_inlier_set(xs,yns,threshold,pose,best_inliers);

        // timer.toc();

        if(inliers>best_inliers){
            //std::cout<<"inliers: "<<inliers<<std::endl;
            best_inliers=inliers;
            best_pose=pose;
# if 0

            // recompute only when neccessary its expensive...
            //double inlier_estimate=best_inliers/((double)xs.size());
            //iters=params.get_iterations(inlier_estimate);

            // perform early exit if exit criteria met
            if( false &&    params.early_exit &&
                    i>params.early_exit_min_iterations &&
                    best_inliers>params.early_exit_inlier_ratio*xs.size()
                    ) break;
#endif
            if(i>100 && best_inliers/double(xs.size())>0.8 && best_inliers>50) break;
        }
        //timer.toc();

    }


    total_iters=i;
    if(best_inliers/double(xs.size())<0.5)
        cout << "pnp total_iters: " << i << " inlier ratio: " << best_inliers/double(xs.size()) <<" inliers "<<best_inliers<< endl;
    if(best_inliers<4) {
        mlog()<< "pnp failure: "<<best_inliers<<"\n";
        return PoseD::Identity();
    }
    best_pose=refine(xs, yns, best_pose);

    return best_pose; // will be identity if a complete failure...
}



inline bool inlier(const PoseD& P,const Vector3d& x_w,const Vector2d& yn, double threshold_squared)
{
    auto x_c=P*x_w;
    if(x_c[2]<1e-6) return false;
    return (x_c.dehom() - yn).squaredNorm()<threshold_squared;
}

void refine_from(const std::vector<Vector3d>& xs,
                 const std::vector<Vector2d>& yns,
                 PoseD& P, double threshold_squared)
{
    //mlog()<<"samples: "<<xs.size()<<"\n";
    std::vector<Vector3d> inlier_xs;inlier_xs.reserve(xs.size());
    std::vector<Vector2d> inlier_yns;inlier_yns.reserve(xs.size());
    ceres::Problem problem;
    ceres::LossFunction* loss=nullptr;// implies squared
    for(int i=0;i<xs.size(); ++i){
        if(!inlier(P, xs[i],yns[i],threshold_squared)) continue;
        inlier_xs.push_back(xs[i]);
        inlier_yns.push_back(yns[i]);
    }

    if(inlier_xs.size()>3)
    {
        problem.AddResidualBlock(PnPReprojectionError::Create(inlier_xs,inlier_yns),nullptr,P.getRRef(),P.getTRef());

        ceres::LocalParameterization* qp = new ceres::QuaternionParameterization;
        problem.SetParameterization(P.getRRef(), qp);

        ceres::Solver::Options options;{
            options.linear_solver_type = ceres::DENSE_SCHUR;
            options.num_threads=std::thread::hardware_concurrency();
            options.max_num_iterations=5;
        }
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        //cout<<"Report 1: \n"<<summary.FullReport()<<endl;
    }
    else
    {
        mlog()<<"too few inliers in pnp refine: "<<inlier_xs.size()<<"\n";
        exit(1);
    }
}

/**
 * @brief PNP::refine
 * since we expect a high noise, low outlier ratio solution(<50%), we should refine using a cutoff loss twice...
 */
PoseD PNP::refine(const std::vector<Vector3d>& xs,/// the known 3d positions in world coordinates
                  const std::vector<Vector2d>& yns,/// the pinhole normalized measurements corresp to xs
                  PoseD best_pose)
{

    double thr=params.threshold->value();
    thr*=thr;
    refine_from(xs,yns,best_pose,thr);  
    refine_from(xs,yns,best_pose,thr);


    return best_pose;
}

} // end namespace cvl

