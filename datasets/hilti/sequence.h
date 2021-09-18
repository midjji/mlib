#pragma once
using uint=unsigned int;
//#include <mlib/datasets/hilti/calibration.h>
#include <mlib/datasets/hilti/sample.h>


namespace cvl{
namespace hilti {


class Sequence
{

public:
    using sample_type=std::shared_ptr<Sample>;
    Sequence(std::string path);



    int samples() const;
    std::shared_ptr<Sample> sample(int index) const;
    int rows() const;
    int cols() const;
    std::string name() const;
    std::vector<double> times() const ;
    int sequence_id() const ;
    std::string path;



private:
    // the samples are sequential, with no missing? yes,

    std::map<int, float128> index2time;
    std::map<float128, std::map<int, std::string>> image_paths;

    void prepare_image_paths();
};





}
}
