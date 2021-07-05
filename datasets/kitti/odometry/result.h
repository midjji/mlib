#pragma once
#include <mlib/datasets/kitti/odometry/kitti.h>



namespace cvl{
namespace kitti{


/**
 * @brief The KittiError class
 *  computes the exact error mandated by the kitti benchmark
 */
class KittiError
{
public:
    KittiError(cvl::PoseD deltaPose,double len, double speed);
    double   r_err;     // the erorr in kittirotationerror? quaternion angle difference?
    double   t_err;     // the error in meters
    double   len;   // the length of the interval
    double   speed; // the average speed travelled during the interval
    double kittiRotationError(PoseD err);
};

/**
 * @brief The DistLsh class
 * KITTI specific distance hashing to speed up length quires
 */
class DistLsh{
public:
    DistLsh(){}
    ~DistLsh(){}
    DistLsh(std::vector<cvl::PoseD> ps);
    /**
     * @brief getIndexPlusDist returns the index of the frame which is atleast dist after start but atmost dist+2m or -1 if no such frame is found
     * @param start
     * @param dist
     * @return
     */
    int getIndexPlusDist(uint start, double dist) const;
private:
    std::vector<double> dists;
    std::vector<std::vector<std::pair<uint,double>>> distmap;
};
/**
 * @brief getDistanceTraveled
 * @param ps
 * @return a vector listing how far the car has travelled at time t
 *
 *  distance[i]= sum(|translation(t) -translation(t-1)|)
 *
 * Assumes transform:
 *
 * X_w=P_wc*X_c ie the inverse of the usual
 *
 * Note this function is used on the gt poses to create the sampling for the eval
 */
std::vector<double> getDistanceTraveled(const std::vector<cvl::PoseD>& ps);








/**
 * @brief The kitti:Result class
 *
 * contains all the results relevant for evaluation.
 * Make one of these. Name it, Serialize it and read it back.
 *
 */
class Result{
public:
    Result(){}
    ~Result(){}
    Result(Sequence seq){
        this->seq=seq;
    }

    bool init(std::string path);
    std::string getDisplayString();


    void evaluate();

    void save_evaluation(std::string path);

    std::string state="Uninitialized";

    DistLsh getDistLsh(){
        if(!distlsh_inited)
            distlsh=DistLsh(seq.gt_poses);
        distlsh_inited=true;
        return distlsh;
    }

    std::vector<double> lengths={100,200,300,400,500,600,700,800};



    std::vector<double> getTL( std::vector<double> lengths, double thr=1) const;
    std::vector<double> getRL( std::vector<double> speeds, double thr=1)  const;
    std::vector<double> getTS( std::vector<double> lengths, double thr=2) const;
    std::vector<double> getRS( std::vector<double> speeds, double thr=2)  const;

    void getDistributions(std::vector<double>& interframe_angle_error, std::vector<double>& interframe_translation_error) const{interframe_translation_error=this->interframe_translation_error;interframe_angle_error=this->interframe_angle_error;}



    std::string toMatlabStruct();
    std::string name(){return seq.name();}
    Sequence seq;
    std::vector<PoseD> poses;// in kitti direction
private:





    DistLsh distlsh;
    bool inited=false;
    bool evaluated=false;
    bool distlsh_inited=false;
    int step_size=10; // should be 1 for proper eval but 10 is faster for testing...

    void compute_benchmark_metrics();
    void save_benchmark_metrics(std::string path);
    std::vector<KittiError> kes;


    std::vector<double> interframe_angle_error, interframe_translation_error;
    std::vector<double> interframe_delta_prediction_angle_error, interframe_delta_prediction_translation_error;

};

}// end kitti namespace
}// end namespace cvl

