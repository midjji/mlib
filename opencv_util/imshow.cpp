#include <opencv2/highgui.hpp>
#include <mlib/opencv_util/imshow.h>
#include <mlib/opencv_util/cv.h>
#include <mlib/utils/mlog/log.h>
namespace cvl{
bool imshow(cv::Mat im, std::string name){
    cv::namedWindow(name,cv::WINDOW_GUI_EXPANDED);

    // check if the image is good
    if(im.rows==0 || im.cols==0 ||im.data==nullptr){
        mlog()<<"empty image: "<<name<<std::endl;
        return false;
    }

    // convert to Mat1b or Mat3b if needed

    if(     im.type()==CV_8U ||
            im.type()==CV_16U||
            im.type()==CV_32S||
            im.type()==CV_16F||
            im.type()==CV_32F||
            im.type()==CV_64F)
        im.convertTo(im, CV_8U);

    if(     im.type()==CV_16UC3 ||
            im.type()==CV_16FC3 ||
            im.type()==CV_32FC3 ||
            im.type()==CV_64FC3)
        im.convertTo(im, CV_8UC3);

    if(!(im.type()==CV_8U || im.type()==CV_8UC3)){
        mlog()<<"unknown image format"<<std::endl;
        return false;
    }    
    cv::imshow(name,im);
    return true;
}
bool imshow(std::string name, cv::Mat im){
    return imshow(im,name);
}
char wait(double time/*0 means inf...*/){
    return cv::waitKey(time);
}
void input_window(){
    cv::Mat1b im(100,100,uchar(0));
    imshow("input window", im);
}
}
