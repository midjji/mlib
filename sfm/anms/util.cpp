#include <mlib/sfm/anms/util.h>
#include <opencv2/highgui.hpp>

namespace cvl{
namespace anms{

cv::Mat3b drawData(std::vector<anms::Data>& datas){
    cv::Mat3b im=cv::Mat3b::zeros(1000,1000);
    auto strs=getStrengths(datas);
    float minstr=mlib::min(strs);
    float maxstr=mlib::max(strs);


    for(auto d:datas)
        drawCircle(im,d.y,mlib::Color::codeDepthRedToDarkGreen(d.str,minstr,maxstr),3,3);
    return im;
}
void show(std::vector<Data>& datas,std::string name){
    cv::Mat3b im=drawData(datas);
    cv::imshow(name,im);

}








} // end namespace anms
}// end namespace cvl

