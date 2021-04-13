#pragma once
#include <string>
#include <fstream>
#include <vector>
#include <mlib/utils/string_helpers.h>
#include <mlib/utils/cvl/matrix.h>


namespace cvl {
std::string mots_calib_path(std::string basepath, bool training, int sequence);


struct KittiMotsStereoCalibration{
    double fx,fy,px,py, baseline;
};

class KittiMotsCalibration
{
public:
    static KittiMotsStereoCalibration get_color_stereo(std::string basepath,
                                                       bool training,
                                                       int sequence);







private:
    std::vector<double> p0s,p1s,p2s,p3s,r0s,tvs,trs;

    std::string read_chars(std::ifstream& ifs, int num);
    void read_check_test(std::ifstream& ifs, std::string expected, std::string path);
    std::vector<double> read_doubles(std::ifstream& ifs, int num);

    bool test_sanity(std::vector<double> ds);
    bool test_common_K();

    // not satisfied!
    bool shared_k(KittiMotsCalibration km2);

    bool test_common_translations(KittiMotsCalibration km2);
    Vector3d translation(std::vector<double> ds){
        return Vector3d(ds[3],ds[7],ds[11]);
    }
        bool read_file(std::string path);
        double fx(){        return p0s[0];    }
        double fy(){        return p0s[5];    }
        double px(){return p0s[3];}
        double py(){return p0s[7];}
        double color_baseline(){
            // it varies a fair bit and is in 3d,  and the cameras may change rotation too, and ...
            // perhaps try like this to start with, then refactor the images later.
            return 0.384381;
        }

};




}
