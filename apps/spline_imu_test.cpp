#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <QApplication>
#include <assert.h>
#include <ceres/ceres.h>
#include <ceres/loss_function.h>

#include <mlib/utils/mlibtime.h>

#include <mlib/utils/cvl/matrix.h>

#include <mlib/vis/mlib_simple_point_cloud_viewer.h>

#include <mlib/utils/spline_pose_trajectory.h>
#include <mlib/utils/simulator_helpers.h>
#include <mlib/plotter/plot.h>

#include <mlib/utils/smooth_trajectory.h>
#include <mlib/utils/imu_simulator.h>


using namespace cvl;

using std::cout;using std::endl;

template<class St,int Order>
std::vector<std::vector<PoseD>> show_ps(St gt_spline,
                                        PoseSpline<Order> spline,
                                        int num=1000, int offset=50)
{
    std::vector<std::vector<PoseD>> sps;
    sps.push_back(gt_spline.display_poses());
    sps.push_back(spline.display_poses(gt_spline.interior_times(num,offset)));
    return sps;
}


double mean(std::vector<double>& xs){
    double r=0;
    for(double x:xs)
        r+=x;
    return r/xs.size();
}

template<int Degree0,int Degree> void
view_errors(const PoseSpline<Degree0>& gt_spline,
                            const PoseSpline<Degree>& spline)
{

    std::vector<mlib::Color> cols={mlib::Color::blue(),mlib::Color::green()};
    mlib::pc_viewer("uninitialized regularized incremental estimated spline")->setPointCloud(show_ps(gt_spline,spline),cols);
    std::map<std::string, std::vector<double>> errs;


    std::vector<double> ts;
    for(auto d:gt_spline.interior_times(1000,50)){
        ts.push_back(d);

        // first compare poses!
        PoseD P=spline(d);
        PoseD P_gt=gt_spline(d);

        errs["angle error"].push_back(P.angleDistance(P_gt)*180.0/3.14159265359);
        auto qs=spline.qdot(d);
        auto gtqs=gt_spline.qdot(d);

        errs["q error"].push_back(qs[0].geodesic(gtqs[0].q));
        errs["q' error"].push_back(sign_compensated_error<4,double>(qs[1].q,gtqs[1].q).norm());
        errs["q'' error"].push_back(sign_compensated_error<4,double>(qs[2].q,gtqs[2].q).norm());

        //errs["bpes"].push_back((P.t - P_gt.t).norm());
        errs["t error"].push_back((spline.translation(d,0)-gt_spline.translation(d,0)).norm());
        errs["t' error"].push_back((spline.translation(d,1)-gt_spline.translation(d,1)).norm());
        errs["t'' error"].push_back((spline.translation(d,2)-gt_spline.translation(d,2)).norm());

    }
    for(auto [t, err]:errs) cout<<"mean "<<t<<": "<<mean(err)<<endl;
    plot(ts,errs);



}

template<int Degree0,int Degree> void
estimate_spline_incremental(PoseSpline<Degree0>& gt_spline,
                            PoseSpline<Degree>& spline,
                            bool smoothness_costs=true,
                            bool add_noise=true)
{

    // create samples
    std::vector<Vector6d> imus;
    std::vector<double> ts;
    uint N=50000;
    for(auto time:gt_spline.interior_times(N,0))
    {

        imus.push_back(gt_spline.imu_camera2world(time,false,0,0));
        ts.push_back(time);
    }




    for(uint n=5000;n<N+1;n+=5000)
    {


        ceres::Problem problem;
        std::set<double*> paramss;
        for(uint i=0;i<n && i<imus.size();++i)
        {
            Vector6d imu=imus[i];
            double time=ts[i];
            double weight=1;

            auto tmp= spline.imu_observation_residual_block(time,weight,false, imu);
            auto cost=std::get<0>(tmp);
            auto params=std::get<1>(tmp);
            for(double* d:params)            paramss.insert(d);
            problem.AddResidualBlock<Degree+1>(cost,nullptr,params);

        }

        if(smoothness_costs &&false){

            // get the times for the earliest and latest measurements,
            // then add N knots to each end, or as much as you wanna be able to interpolate
            int first_cpt=spline.get_first(ts[0]) - Degree -200;
            int last_cpt=spline.get_last(ts[n])   + Degree +200;
            int N= last_cpt - first_cpt;
            // this gives the
            auto ts=spline.interior_times_for_control_point_interval(first_cpt, last_cpt, N*20);
            double weight=0.1;
            for(double time:ts) {

                //auto [cost, params] = spline.angular_accelleration_regularization_residual_block(time,weight);
                {
                    auto tmp = spline.angular_accelleration_regularization_residual_block(time,weight);
                    auto cost=std::get<0>(tmp);
                    auto params=std::get<1>(tmp);
                    for(double* d:params) paramss.insert(d);

                    problem.AddResidualBlock<Degree+1>(cost,nullptr,params);
                }
                {
                    auto tmp = spline.accelleration_regularization_residual_block(time,weight);
                    auto cost=std::get<0>(tmp);
                    auto params=std::get<1>(tmp);
                    for(double* d:params) paramss.insert(d);

                    problem.AddResidualBlock<Degree+1>(cost,nullptr,params);
                }
            }
        }


        for(double* param:paramss){
            problem.SetParameterization(param, new ceres::ProductParameterization(
                                            new ceres::QuaternionParameterization(),
                                            new ceres::IdentityParameterization(3)));
        }

        ceres::Solver::Options options;
        {
            options.linear_solver_type = ceres::SPARSE_SCHUR;// does not appear to significantly reduce costs
            options.max_num_iterations=6;
            options.num_threads=std::thread::hardware_concurrency();
        }
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        cout<<"ba Full Summary: \n"<<summary.FullReport()<<endl;
        view_errors(gt_spline, spline);
    }


    //for(int i=0;i<100;++i)        cout<<gt_spline.control_point_implied(i) -spline.control_point_implied(i)<<endl;
    //mlog()<<gt_spline.display()<<endl;
    mlog()<<spline.display()<<endl;
}





int main()
{


//    QApplication app(argc, argv);

    auto gt_spline=default_trajectory_camera2world();
    double delta_time=1;

    view_errors(gt_spline,gt_spline);


    PoseSpline<4> spline(delta_time);
    //estimate_spline_incremental(gt_spline, spline,true,true);

  //  app.exec();
    mlib::sleep(100);
    mlib::wait_for_viewers();

    return 1;
}

