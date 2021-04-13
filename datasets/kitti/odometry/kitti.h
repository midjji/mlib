#pragma once
#include <fstream>
#include <opencv2/core.hpp>
#include <mlib/utils/cvl/pose.h>
#include <mlib/datasets/kitti/odometry/sequence.h>

namespace cvl{

/**
 * @brief The KittiDataset class
 * linux only convenient wrapper for the standard kitti data,
 *
 */
namespace kitti{






class KittiOdometrySample{
public:
    std::vector<cv::Mat1w> images;
    cv::Mat1f disparity;
    int sequenceid_;
    int frameid_;
    KittiOdometrySample(std::vector<cv::Mat1w> images,
                        cv::Mat1f disparity,
                        int sequenceid_,
                        int frameid_):
        images(images),
        disparity(disparity),
        sequenceid_(sequenceid_),
        frameid_(frameid_)
    {}
    int frameid(){return frameid_;}
    int sequenceid(){return sequenceid_;}
    uint rows(){return disparity.rows;}
    uint cols(){return disparity.cols;}
    cv::Mat1b disparity_image(); // for visualization, new clone
    cv::Mat3b disparity_image_rgb(); // for visualization, new clone
    cv::Mat3b rgb(uint id);// for visualization, new clone
    cv::Mat1w grey(uint id);// for visualization, new clone
    cv::Mat1b greyb(uint id);// for visualization, new clone
    cv::Mat1f greyf(uint id);// for visualization, new clone

};





/**
 * @brief The KittiDataset class
 *
 * Contains all info relating to the kitti dataset,
 * Creating a instance of this class allows verifying the kitti dataset is complete,
 *  and allows convenient access to the pertinent data
 */
class KittiDataset{
public:
    using sample_type=std::shared_ptr<KittiOdometrySample>;
    KittiDataset(){}
    ~KittiDataset(){}
    KittiDataset(std::string basepath);
    void init();

    bool checkFiles();


    std::string getseqpath(int sequence);
    bool getImage(std::vector<cv::Mat1b>& images, int number, int sequence, bool cycle=false);

    Sequence getSequence(unsigned int index){init();        return seqs.at(index);    }
    std::shared_ptr<KittiOdometrySample> get_sample(int sequence, int frameid);



    int index=0;
    int sequence=0;
    std::shared_ptr<KittiOdometrySample> next();


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



    // Fixed data!

    /// the sequence indexes
    std::vector<int> sequences={0,       1,    2,    3,    4,    5,    6,    7,    8,    9,   10,   11,  12,   13,    14,   15,   16,   17,   18,   19,   20,   21};
    /// the number of images per sequence( to verify the database is intact)
    std::vector<int> seqimgs=  {4541, 1101, 4661,  801,  271, 2761, 1101, 1101, 4071, 1591, 1201,  921, 1061, 3281,  631, 1901, 1731,  491, 1801, 4981,  831, 2721};
    /// the number of image rows in each sequence
    std::vector<int> rowss=    {376,  376,  376,   375,  370,  370,  370,  370,  370,  370,  370,  370,  370,  376,  376,  376,  376,  376,  376,  376,  376,  376};
    /// the number of image cols in each sequence
    std::vector<int> colss=    {1241, 1241, 1241, 1242, 1226, 1226, 1226, 1226, 1226, 1226, 1226, 1226, 1226, 1241, 1241, 1241, 1241, 1241, 1241, 1241, 1241, 1241};

    /// the sequence data
    std::vector<Sequence> seqs;



    // evaluation stuff
    std::vector<double> getEvalLengths(){
        std::vector<double> len={100,200,300,400,500,600,700,800};
        return len;
    }





private:
    bool inited=false;
};




void testKitti(std::string basepath="/store/datasets/kitti/dataset/", bool withstereo=false);









}// end kitti namespace
}// end namespace cvl
