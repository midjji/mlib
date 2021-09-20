#include <mlib/datasets/stereo_sample.h>
#include <mlib/opencv_util/imshow.h>
#include <mlib/opencv_util/convert.h>

namespace cvl{

StereoSample::StereoSample(float128 time,const StereoSequence* ss,
             int frame_id,
                           std::vector<cv::Mat1f> images,
             cv::Mat1f disparity_):
    ImageSample(time, ss, frame_id,images),
    disparity_(disparity_)
{}

cv::Mat1f StereoSample::disparity_image() const{return disparity_.clone();}



StereoSample::~StereoSample(){}
cv::Mat3b StereoSample::display_disparity() const
{
    return image2rgb3b(disparity_,2);
}
float StereoSample::disparity(double row, double col) const{
    if(row<0) return -1.0F;
    if(col<0) return -2.0F;
    if(rows()<=row) return -3.0F;
    if(cols()<=col) return -4.0F;
    if(std::isinf(row+col)) return -5.0F;
    if(std::isnan(row+col)) return -6.0F;
    float d=disparity(std::round(row),std::round(col));
    if(!std::isnormal(d)) return -7.0F;
    return d;
}
int StereoSample::type() const{
    return Sample::stereo;
}
float StereoSample::disparity(Vector2d rowcol)const{return disparity(rowcol[0],rowcol[1]);}

void show(const std::shared_ptr<StereoSample>& s){
    imshow(s->rgb(0), "stereo sample left");
    imshow(s->rgb(1), "stereo sample right");
    imshow(s->display_disparity(), "disparity*2");
}

}
