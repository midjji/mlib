#pragma once
#include <map>
#include <atomic>
#include <opencv4/opencv2/core.hpp>
#include <mlib/utils/cvl/pose.h>
#include <mlib/utils/bounding_box.h>


namespace cvl{


class KittiMotsSample{
public:


    // dataset contains rgb3b images
    KittiMotsSample(std::map<std::string,std::string> paths,
                    bool training,
                    int sequence,
                    int frameid):
        paths(paths),training(training),
        sequence(sequence),frameid(frameid){loaded=false;load_all();}
    void load_all();
    cv::Mat3b rgb(uint id);// for visualization, new clone
    cv::Mat1b greyb(uint id);
    cv::Mat1w greyw(uint id);



    std::map<std::string,std::string> paths;
    bool training;
    int sequence;
    int frameid;
    cv::Mat3b left,right;
    cv::Mat1b lb,rb;
    cv::Mat1w lw,rw;

    std::mutex mtx;
    std::atomic<bool> loaded;   
};



}
