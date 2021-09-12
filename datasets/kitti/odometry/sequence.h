#pragma once
#include <opencv2/core/mat.hpp>
#include <mlib/utils/cvl/pose.h>
#include <mlib/datasets/stereo_sequence.h>
#include <mlib/datasets/kitti/odometry/sample.h>
#include <mlib/datasets/kitti/odometry/fid2time.h>
#include <mlib/datasets/kitti/odometry/distlsh.h>
namespace cvl{
namespace kitti{

/**
 * @brief The Sequence class
 * contains all info and functions relevant for one kitti sequence!
 */
class Sequence: public StereoSequence
{
public:
    int samples() const override;
    int rows()    const override;
    int cols()    const override;
    StereoCalibration calibration() const override;
    std::string name() const override;
    std::shared_ptr<StereoSample> sample(int index) const override;
    std::shared_ptr<Frameid2TimeMap> fid2time() const override;
    std::vector<double> times() const override;
    int sequence_id() const override;
    std::vector<PoseD> gt_poses() const override;






    int sequence() const;


    std::string description() const;
    double baseline() const;

    std::shared_ptr<KittiOdometrySample> get_sample(int index) const;
    double fps() const;




    Sequence()=default;
    Sequence(std::string path_, int sequence_,int rows_, int cols_, int samples_);


    bool getImages(std::vector<cv::Mat1b>& images, int number) const;
    PoseD getPose(int number) const;
    PoseD gt_pose(int number) const;
    PoseD getPoseRightLeft() const;
    bool getImages(std::vector<cv::Mat1w>& images,cv::Mat1f& disparity, int number) const;


    // this (row, col) = (K*x_cam).dehom();
    cvl::Matrix3d getK() const;
    PoseD P_camera_vehicle() const;




    std::vector<Matrix34d> ks; //ks[0] is the left cam, ks[1] is the right



    bool is_training() const;

    // loop closure related
    std::vector<unsigned int> getDistantFrames();
    cv::Mat1b getPoseConfusionMatrix();
    cv::Mat3b getMap();


    std::string seqpath() const;
    std::shared_ptr<Sequence> shrunk(int newsize=100) const;

    // evaluation
    DistLsh dist_lsh();

private:

    std::vector<double> times_;
    std::vector<PoseD> gt_poses_; // Pwc(t)
    PoseD P_camera_vehicle_; // Including the transform to my coordinates
    PoseD P_cvl_kitti;
    std::string path_;
    int sequence_=-1;
    int rows_=0;
    int cols_=0;
    int samples_=0;
    std::string name_;
    std::string description_;
    // baseline in meters!
    double baseline_; // >0 P10(-Vector3d(baseline,0,0)); x_1=P10*x0 . 0 is left, 1 is right.
    bool inited=false;
    DistLsh distlsh;
};



/** Reading kitti style groundtruth */
/**
 * @brief getPoses Getting kitti poses!
 * @param path
 * @return vector of poses
 *
 *  x_w=P*x_ci where P are 3x4 matrixes, with a implicit row of 0 0 0 1 at the bottom.
 */
std::vector<cvl::PoseD> readKittiPoses(std::string path, bool require_found=true);
/**
 * @brief writeKittiPoses write poses to path in the kitti format, see read poses
 * @param path
 * @param ps P should be in x_w=P*x_ci form
 */
void writeKittiPoses(std::string path, std::vector<PoseD> ps);
/**
 * @brief readCalibrationFile reads the kitti calibration file
 * @param path
 * @return
 */
std::vector<Matrix34d>  readCalibrationFile(std::string path);

/**
 * @brief readTimes reads the time stamp file
 * @param path
 * @param count
 * @return
 */
std::vector<double> readTimes(std::string path);

}// end kitti namespace
}// end namespace cvl
