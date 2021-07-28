
#include <mlib/datasets/stereo_sample.h>
#include <mlib/datasets/daimler/sample.h>
#include <mlib/datasets/kitti/odometry/sample.h>
#include <mlib/opencv_util/imshow.h>

namespace cvl{
StereoSample::~StereoSample(){}
StereoSample::StereoSample(int sample_id,double sample_time, cv::Mat3b left,cv::Mat3b right, cv::Mat1f disparity,
             cv::Mat1f lf,cv::Mat1f rf,int sequence_id):sample_id(sample_id),sample_time(sample_time),
    left(left),right(right),disparity_image(disparity),lf(lf),rf(rf),sequence_id(sequence_id){}
int StereoSample::rows() const{return left.rows;}
int StereoSample::cols() const{return left.cols;}
int StereoSample::id() const{return sample_id;}
double StereoSample::time() const{return sample_time;}

cv::Mat1f StereoSample::get1f(int i){    
    if(i==0)
        return lf;
    return rf;
}
cv::Mat3b StereoSample::rgb(int i){
    if(i==0) return left;
    return right;
}
cv::Mat3b StereoSample::display_disparity()
{
    /*
    cv::Mat1b in=left;
    if(i==1)
        in=right;
    cv::Mat3b out;
    cv::cvtColor(in, out, cv::COLOR_GRAY2BGR);
*/

   auto& in=disparity_image;
   cv::Mat3b out(in.rows,in.cols);
   for(int r=0;r<in.rows;++r)
       for(int c=0;c<in.cols;++c)
       {
           float f=2*in(r,c);
           if(f<0) f=0;
           if(f>=255) f=255;
           uchar v=uchar(f);
           out(r,c)=cv::Vec3b(v,v,v);
            }
   return out;
}
float StereoSample::disparity(int row, int col) const{
    if(row<0) return -1.0f;
    if(col<0) return -2.0f;
    if(disparity_image.rows<=row) return -3.0f;
    if(disparity_image.cols<=col) return -4.0f;
    return disparity_image(row,col);
}
float StereoSample::disparity(double row, double col) const{
    return disparity(int(std::round(row)),
                     int(std::round(col)));
}
float StereoSample::disparity(Vector2d rowcol)const{return disparity(rowcol[0],rowcol[1]);}
int StereoSample::sequenceid() const{
    return sequence_id;
}
std::shared_ptr<StereoSample> convert2StereoSample(std::shared_ptr<DaimlerSample> sd){
    if(!sd) return nullptr;
    auto ret=std::make_shared<StereoSample>(
                sd->frameid(),
                sd->time(),
                sd->rgb(0).clone(),
                sd->rgb(1).clone(),
                sd->disparity_imagef().clone(),
                sd->grayf(0).clone(),
                sd->grayf(1).clone(),sd->sequenceid());
    return ret;
}

std::shared_ptr<StereoSample> convert2StereoSample(std::shared_ptr<kitti::KittiOdometrySample> sd){
    if(!sd) return nullptr;
    auto ret=std::make_shared<StereoSample>(
                sd->frameid(),
                sd->time(),
                sd->rgb(0).clone(),
                sd->rgb(1).clone(),
                sd->disparity_imagef().clone(),
                sd->greyf(0).clone(),
                sd->greyf(1).clone(),sd->sequenceid());
    return ret;
}
void show(const std::shared_ptr<StereoSample>& s){
    imshow(s->rgb(0), "stereo sample left");
    imshow(s->rgb(1), "stereo sample right");
    imshow(s->display_disparity(), "disparity*2");
}

}
