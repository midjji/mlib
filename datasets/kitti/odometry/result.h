#pragma once
#include <mlib/datasets/kitti/odometry/kitti.h>



namespace cvl{
namespace kitti{


/**
 * @brief The KittiError class
 *  computes the exact error mandated by the kitti benchmark
 */
struct KittiError
{
    KittiError()=default;
    KittiError(cvl::PoseD deltaPose,double len, double speed);
    double   r_err;     // the erorr in kittirotationerror? quaternion angle difference?
    double   t_err;     // the error in meters
    double   len;   // the length of the interval
    double   speed; // the average speed travelled during the interval
    double kittiRotationError(PoseD err);

    double t_err_relative() const;
    double r_err_relative() const;
};

std::vector<KittiError> compute_benchmark_metrics(DistLsh& distlsh,
                                                  std::vector<PoseD> gt_poses,
                                                  std::vector<PoseD> poses,
                                                  std::vector<double> lengths);


/**
 * @brief The kitti:Result class
 *
 * contains all the results relevant for evaluation.
 * Make one of these. Name it, Serialize it and read it back.
 *
 */
class Result{
public:
    Result()=default;
    Result(Sequence& seq,
           std::vector<PoseD> Pwcs,
           std::string name,
           std::vector<double> lengths={100,200,300,400,500,600,700,800});
    std::vector<double> lengths; // largely static, defined by the benchmark

    std::string getDisplayString();

    std::map<std::string, std::vector<double>> graphs() const;
    std::vector<double> getTL( std::vector<double> lengths, double thr=1) const;
    std::vector<double> getRL( std::vector<double> speeds, double thr=1)  const;
    std::vector<double> getTS( std::vector<double> lengths, double thr=2) const;
    std::vector<double> getRS( std::vector<double> speeds, double thr=2)  const;

    void getDistributions(std::vector<double>& interframe_angle_error, std::vector<double>& interframe_translation_error) const{interframe_translation_error=this->interframe_translation_error;interframe_angle_error=this->interframe_angle_error;}



    std::string toMatlabStruct();
    std::string name() const;


    double translation_error_average() const;
    double rotation_error_average() const;

    std::string name_;

    void save_benchmark_metrics(std::string path);
    std::vector<KittiError> kes;
    Vector2d mean_errs() const;

    std::vector<double> interframe_angle_error, interframe_translation_error;
    std::vector<double> interframe_delta_prediction_angle_error, interframe_delta_prediction_translation_error;

};

}// end kitti namespace
}// end namespace cvl

