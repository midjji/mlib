#include <mlib/datasets/hilti/sample.h>
#include <mlib/opencv_util/convert.h>
#include <mlib/opencv_util/imshow.h>
#include <mlib/utils/string_helpers.h>

namespace cvl {
namespace hilti {
HiltiImageSample::HiltiImageSample( float128 time,const StereoSequence* ss,
            int  frame_id_, std::map<int,cv::Mat1f> images, std::vector<imu::Data> imu_datas):
    StereoSample( time, ss, frame_id_, std::vector<cv::Mat1f>(), images[5]),
    images(images),imu_datas(imu_datas){}


bool HiltiImageSample::complete() const{for(int i=0;i<6;++i) if(!has(i)) return false;return true;}
bool HiltiImageSample::stereo() const{    return has(0) && has(1) && has(5);}
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

void HiltiImageSample::show() const{

    for(const auto& [id,image]:images) {    
        imshow(image,"hilti cam "+str(id));

    }
}

}
}

