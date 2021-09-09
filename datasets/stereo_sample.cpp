#include <mlib/datasets/stereo_sample.h>
#include <mlib/opencv_util/imshow.h>

namespace cvl{
StereoSample::~StereoSample(){}
cv::Mat3b StereoSample::display_disparity() const
{
    auto in=disparity_image();
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
float StereoSample::disparity(double row, double col) const{
    if(row<0) return -1.0F;
    if(col<0) return -2.0F;
    if(rows()<=row) return -3.0F;
    if(cols()<=col) return -4.0F;
    if(std::isinf(row+col)) return -5.0F;
    if(std::isnan(row+col)) return -6.0F;
    float d=disparity_impl(row,col);
    if(!std::isnormal(d)) return -7.0F;
    return d;
}

float StereoSample::disparity(Vector2d rowcol)const{return disparity(rowcol[0],rowcol[1]);}

void show(const std::shared_ptr<StereoSample>& s){
    imshow(s->rgb(0), "stereo sample left");
    imshow(s->rgb(1), "stereo sample right");
    imshow(s->display_disparity(), "disparity*2");
}

}
