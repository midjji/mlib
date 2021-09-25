#pragma once
using uint=unsigned int;
//#include <mlib/datasets/hilti/calibration.h>
#include <mlib/datasets/hilti/sample.h>

#include <mlib/datasets/stereo_sequence.h>
#include <mlib/datasets/hilti/calibration.h>
#include <mlib/utils/imu_data.h>

namespace cvl{
namespace hilti {






struct PreloadSample
{
    PreloadSample()=default;
    PreloadSample(int sampleindex, float128 time,float128 original_time_ns,
                  std::vector<imu::Data> datas,
                  std::map<int,std::string> cam2paths );

    std::shared_ptr<HiltiImageSample> load(const std::shared_ptr<StereoSequence> ss) const;
    bool has(int index) const;
    bool has_all() const;
    std::string str() const;
    float128 time() const{return time_;}
    float128 original_time_ns() const{return original_time_ns_;}
private:
    int sampleindex;
    float128 time_{0}; // for the images,
    float128 original_time_ns_{0};
    std::vector<imu::Data> datas=std::vector<imu::Data>(); // from image time to next image time. except for the first and last image times.
    std::map<int, std::string> cam2paths{}; // the rectified paths, using standardized numbers
    std::map<std::string, int> name2num{
        {"left", 0},
        {"right", 1},
        {"cam2",2},
        {"cam3",3},
        {"cam4",4},
        {"disparity",5}};


};



class Sequence: public StereoSequence
{
    //hilti/preprocessed/sequence_name/
    //                                 times.txt // all the times around
    //                                 post_rectification_calibration.txt
    //                                 left/{time}.exr   // left stereo rectified, not just nonlinear rectified!
    //                                 right/{time}.exr   // left stereo rectified, not just nonlinear rectified!
    //                                 cam0 symlink to left, ie we swap them
    //                                 cam1 symlink to right
    //                                 cam2/{time}.exr
    //                                 cam3/{time}.exr
    //                                 cam4/{time}.exr
    //                                 disparity/ // this is a symlink to a selected disparity method folder
    //                                 disparity_method0/{time}.exr





    // We change the camera numbers to fit mlib,
    // We use the rectified images and their numbers are:
    std::map<std::string, int> name2num{
        {"left", 0},
        {"right", 1},
        {"cam2",2},
        {"cam3",3},
        {"cam4",4},
        {"disparity",5}};
    std::map<int, std::string> num2name;
    std::map<int, std::string> num2path{
        {0,"left"}, // 0, 1 should be swapped...
        {1,"right"},
        {2,"cam2"},
        {3,"cam3"},
        {4,"cam4"},
        {5,"disparity"}, // should be disparity
    }; // rectified paths
    std::string image_format=".exr"; // should be ".exr" for the rectified and disparity


    Sequence(std::string path, std::string sequence_name);
    std::weak_ptr<Sequence> wself;
public:
static std::shared_ptr<Sequence> create(std::string path, std::string sequence_name);



    std::shared_ptr<StereoSample> stereo_sample(int index) const;
    std::shared_ptr<HiltiImageSample> sample(int index) const;
    int samples() const override;
    int rows() const override;
    int cols() const override;
    std::string name() const override;




    StereoCalibration calibration() const override;
    StereoCalibration calibration(int index) const;
    Calibration hilti_calibration() const;


    std::shared_ptr<Frameid2TimeMap> fid2time() const override;
    virtual std::vector<double> times() const{return std::vector<double>();};







private:
    void read_metadata(std::string path);
    Calibration calib;
    std::shared_ptr<Frameid2TimeMapLive> f2t;
    float128 t0;
    const std::string sequence_name;
    std::vector<PreloadSample> preload_samples;
    std::map<float128, std::map<int, std::string>> load_image_paths(std::string path);
    std::string rectified_path(std::string basepath, int camera, int index) const;
};





}
}
