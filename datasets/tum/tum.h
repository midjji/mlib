#pragma once
/* ********************************* FILE ************************************/
/** \file    tum.h
 *
 * \brief    This header contains the tum dataset and helpers
 *
 * \remark
 *
 *
 *
 *
 * \author   Mikael Persson
 * \date     2017-08-11
 * \note MIT licence
 *
 ******************************************************************************/

#include <opencv2/core.hpp>
#include <mlib/utils/cvl/pose.h>


namespace cvl{



std::vector<std::pair<PoseD,PoseD>>
read_tum_poses(std::string path);

class TumTimePoint{
public:
    int index;
    PoseD P0w;
    PoseD P1w;
    Matrix3d K; // all in common

    // 0 is left, 1 is right
    cv::Mat3b rgb0,rgb1;
    cv::Mat1f gray0,gray1;
    cv::Mat1f disp01,disp10; // direction
    cv::Mat1f object_left,object_right; // direction

    void getImages(std::vector<cv::Mat1w>& images);

};

class TumSequence{
public:
    std::vector<std::pair<PoseD,PoseD>> poses_gt;
    TumSequence(std::string drivingpath,
                std::string speed,
                std::string direction,
                std::string focallength,Matrix3d K);
    ~TumSequence(){}
    uint rows=540;uint cols=960;

    std::string drivingpath;
    std::string speed="slow/";
    std::string direction="scene_forwards/";
    std::string focallength="35mm_focallength/";
    double baseline=1; // THE POSES ARE WIERD BUT THE LENGTH IS 1
    Matrix3d K;
    Matrix3d getK();

    TumTimePoint getTimePoint(uint index);
    uint size=0;



};

/**
 * @brief The TumDataset class
 *
 * There are numerous sequences, but lets start with the driving one...
 *
 * Its camera_data/15,35/fast,slow/left,right
 * so 8 of them...
 *
 */
class TumDataset{
public:
    std::string basepath;
    std::vector<TumSequence> sequences;

    TumDataset();

    TumDataset(std::string basepath);

    void init();

    TumSequence getSequence(uint sequence);


    /*
     *
    The virtual imaging sensor has a size of 32.0mmx18.0mm.
    Most scenes use a virtual focal length of 35.0mm. For those scenes, the virtual camera intrinsics matrix is given by
    fx=1050.0	0.0	cx=479.5
    0.0	fy=1050.0	cy=269.5
    0.0	0.0	1.0
    where (fx,fy) are focal lengths and (cx,cy) denotes the principal point.

    Some scenes in the Driving subset use a virtual focal length of 15.0mm (the directory structure describes this clearly). For those scenes, the intrinsics matrix is given by
    fx=450.0	0.0	cx=479.5
    0.0	fy=450.0	cy=269.5
    0.0	0.0	1.0

    */
    Matrix3d K35=Matrix3d(1050,0,479.5,
                          0,1050,269.6,
                          0,0,1);
    Matrix3d K15=Matrix3d(450,0,479.5,
                          0,450,269.6,
                          0,0,1);

};
}
