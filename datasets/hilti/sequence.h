#pragma once
using uint=unsigned int;
//#include <mlib/datasets/hilti/calibration.h>
#include <mlib/datasets/hilti/sample.h>

#include <mlib/datasets/stereo_sequence.h>
#include <mlib/datasets/hilti/calibration.h>
#include <mlib/utils/imu_data.h>

namespace cvl{
namespace hilti {






struct PreloadSample{
    float128 time; // for the images,
    std::vector<imu::Data> datas; // from image time to next image time. except for the first and last image times.
    std::map<int, std::string> paths; // the rectified paths, using standardized numbers
    std::map<std::string, int> name2num{
        {"left", 0},
        {"right", 1},
        {"cam2",2},
        {"cam3",3},
        {"cam4",4},
        {"disparity",5}};
    std::shared_ptr<HiltiImageSample> load(int sampleindex, const StereoSequence* ss) const;
};



class Sequence: public StereoSequence
{
    // unless something weird happens we have a common calibration for all cameras after rectification.

    double fy,fx,py,px;
    int rows_=1080;
    int cols_=1440;
    double baseline=0.11;

    double imu_bias=0;/// osv...


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
    std::map<int, std::string> num2path; // rectified paths



public:

    Sequence(std::string path, std::string sequence_name);

    std::shared_ptr<StereoSample> sample(int index) const override;
    int samples() const override;
    int rows() const override;
    int cols() const override;
    std::string name() const override;


    StereoCalibration calibration() const override;
    Calibration hilti_calibration() const;


    std::shared_ptr<Frameid2TimeMap> fid2time() const override;
    virtual std::vector<double> times() const{return std::vector<double>();};







private:
    std::shared_ptr<Frameid2TimeMapLive> f2t;
    float128 t0;
    const std::string sequence_name;
    std::vector<PreloadSample> preload_samples;
    std::map<float128, std::map<int, std::string>> load_image_paths(std::string path);
    std::string rectified_path(std::string basepath, int camera, int index) const;
};





}
}
