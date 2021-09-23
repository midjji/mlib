#pragma once

#include <opencv2/core/mat.hpp>
#include <mlib/utils/cvl/pose.h>
#include <mlib/datasets/stereo_dataset.h>
#include <mlib/datasets/kitti/odometry/sequence.h>


namespace cvl{

/**
 * @brief The KittiDataset class
 * linux only convenient wrapper for the standard kitti data,
 *
 */
namespace kitti{


/**
 * @brief The KittiDataset class
 *
 * Contains all info relating to the kitti dataset,
 * Creating a instance of this class allows verifying the kitti dataset is complete,
 *  and allows convenient access to the pertinent data
 */
class KittiDataset : public StereoDataset
{


public:
    using sample_type=std::shared_ptr<KittiOdometrySample>;
    KittiDataset()=default;
    KittiDataset(const std::string& basepath);

    std::vector<std::shared_ptr<StereoSequence>> sequences() const override;


    bool checkFiles();


    std::string getseqpath(int sequence);
    bool getImage(std::vector<cv::Mat1b>& images, int number, int sequence, bool cycle=false);

    std::shared_ptr<Sequence> getSequence(int index);




    int index=0;
    int sequence=0;
    // not always true, but always returned by this thing
    int training_sequences=11;






    int images(int sequence);

    std::string basepath;
    /* // K varies per sequence!
    cvl::Matrix3<double> K=cvl::Matrix3<double>(718.856, 0, 607.1928,
                                                    0, 718.856, 185.2157,
                                                    0, 0, 1 );
    */
    // offset to second camera, its on the right... varies per sequence?
    //double baseline=(3.861448/7.18856);
    //cvl::PoseD P10(Vector3d(-baseline));


    /// the sequence data
    std::vector<std::shared_ptr<Sequence>> seqs;


    std::vector<std::shared_ptr<Sequence>> get_training_sequences();
    // Fixed data!
    /// the sequence indexes
    std::vector<int> sequence_indexes{0,       1,    2,    3,    4,    5,    6,    7,    8,    9,   10,   11,  12,   13,    14,   15,   16,   17,   18,   19,   20,   21};
    /// the number of images per sequence( to verify the database is intact)
    std::vector<int> seqimgs  {4541, 1101, 4661,  801,  271, 2761, 1101, 1101, 4071, 1591, 1201,  921, 1061, 3281,  631, 1901, 1731,  491, 1801, 4981,  831, 2721};
    /// the number of image rows in each sequence
    std::vector<int> rowss    {376,  376,  376,   375,  370,  370,  370,  370,  370,  370,  370,  370,  370,  376,  376,  376,  376,  376,  376,  376,  376,  376};
    /// the number of image cols in each sequence
    std::vector<int> colss    {1241, 1241, 1241, 1242, 1226, 1226, 1226, 1226, 1226, 1226, 1226, 1226, 1226, 1241, 1241, 1241, 1241, 1241, 1241, 1241, 1241, 1241};
    // for the benchmark
    std::vector<double> eval_lengths{100,200,300,400,500,600,700,800};

};

// there really only ever needs to be one of these!
const KittiDataset& dataset(std::string path="/storage/datasets/kitti/odometry/");



std::vector<std::vector<PoseD>> trajectories(std::string basepath="/store/datasets/kitti/dataset/");






}// end kitti namespace
}// end namespace cvl
