#include <mlib/datasets/hilti/sample.h>
#include <mlib/opencv_util/convert.h>
#include <mlib/opencv_util/imshow.h>
#include <mlib/utils/string_helpers.h>
#include <mlib/datasets/hilti/sequence.h>

namespace cvl {
namespace hilti {
cv::Mat1f disp(const std::map<int,cv::Mat1f>& images){

    auto it=images.find(5);
    if(it==images.end()) return cv::Mat1f(1080,1440,-1.0f);
    return it->second;
}


HiltiImageSample::HiltiImageSample(
        float128 time,const std::shared_ptr<StereoSequence>ss,
        int  frame_id_, const std::map<int,cv::Mat1f> images, std::vector<imu::Data> imu_datas,
        float128 original_time):
    StereoSample( time, ss, frame_id_, std::vector<cv::Mat1f>(), disp(images)),
    images(images),imu_datas(imu_datas),original_time_ns(original_time)
{







    for(const auto& [ind, image]:images)
    {
        require(image.rows>0," zero rows for image? ?");
        require(image.cols>0," zero cols for image??");
        require(image.data!=nullptr," image data is nullptr?");    }
}


bool HiltiImageSample::complete() const{for(int i=0;i<6;++i) if(!has(i)) return false;return true;}
bool HiltiImageSample::has_stereo() const{    return has(0) && has(1) && has(5);}
bool HiltiImageSample::has(int i) const{    auto it=images.find(i);    return it!=images.end();}

cv::Mat1f HiltiImageSample::grey1f(int i) const
{
    auto it=images.find(i);
    if(it==images.end())
    {
        mlog()<<"bad index: "<<i<<"\n";
        exit(1);
    }

    return it->second.clone();
}

int HiltiImageSample::type() const{    return Sample::hilti;}


std::string HiltiImageSample::num2name(int num) const{
    switch (num) {
    case 0: return "left";
    case 1: return "right";
    case 2: return "cam"+str(num);
    case 3: return "cam"+str(num);
    case 4: return "cam"+str(num);
    case 5: return "disparity";
    default:wtf(); return "wtf!";
    }
}
std::shared_ptr<Sequence> HiltiImageSample::hilti_sequence() const
{
    return std::dynamic_pointer_cast<Sequence>(sequence());
}
int HiltiImageSample::rows() const {return 1080;}
int HiltiImageSample::cols() const {return 1440;}

}
}

