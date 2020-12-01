#pragma once
#include <mlib/utils/cvl/pose.h>
#include <opencv2/core.hpp>
namespace cvl{
namespace kitti{

/**
 * @brief The Sequence class
 * contains all info and functions relevant for one kitti sequence!
 */
class Sequence{
public:
    Sequence(){}
    ~Sequence(){}
    Sequence(std::string path_, int sequence_,int rows_, int cols_, int images_);
    void readSequence();

    bool getImages(std::vector<cv::Mat1b>& images, int number) const;
    PoseD getPose(int number) const;
    PoseD getPoseRightLeft() const;
    bool getImages(std::vector<cv::Mat1w>& images,cv::Mat1f& disparity, int number) const;




    cvl::Matrix3d getK();


    std::string path,seqpath;
    int rows, cols, images,sequence;
    std::string name,description;
    // baseline in meters!
    double baseline; // >0 P10(-Vector3d(baseline,0,0)); x_1=P10*x0 . 0 is left, 1 is right.

    std::vector<Matrix34d> ks; //ks[0] is the left cam, ks[1] is the right
    // in seconds from start
    std::vector<double> times;
    std::vector<PoseD> gt_poses; // Pwc(t)

    bool isTraining();

    // loop closure related
    std::vector<unsigned int> getDistantFrames();
    cv::Mat1b getPoseConfusionMatrix();
    cv::Mat3b getMap();



private:


};

/** Reading kitti style groundtruth */
/**
 * @brief getPoses Getting kitti poses!
 * @param path
 * @return vector of poses
 *
 *  x_w=P*x_ci where P are 3x4 matrixes, with a implicit row of 0 0 0 1 at the bottom.
 */
std::vector<cvl::PoseD> readKittiPoses(std::string path);
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
