#include <thread>
#include <future>
#include <kitti/mots/sample.h>
#include <mlib/utils/mlog/log.h>
 
#include <mlib/utils/cvl/triangulate.h>
#include <experimental/filesystem>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>

namespace fs = std::experimental::filesystem;
using std::cout;
using std::endl;
namespace cvl {




cv::Mat3b KittiMotsSample::rgb(uint id){
    load_all();
    if(id==0) return left.clone();
    return right.clone();

} // for visualization, new clone
cv::Mat1b KittiMotsSample::greyb(uint id){
    load_all();if(id==0) return lb.clone(); return rb.clone();}
cv::Mat1w KittiMotsSample::greyw(uint id){
    load_all();if(id==0) return lw.clone(); return rw.clone();}










cv::Mat3b safe_imread(std::string path){
    if(!fs::exists(fs::path(path)))
        cout<<"image not found: "+ path<<endl;
    cv::Mat3b rgb=cv::imread(path,cv::IMREAD_UNCHANGED);
    return rgb;
}
cv::Mat1w rgb3b2gray16(cv::Mat1b lg){
    cv::Mat1w lw(lg.rows,lg.cols);
    for(int row=0;row<lg.rows;++row)
        for(int col=0;col<lg.cols;++col){
            lw(row,col)=lg(row,col)*16;
        }
    return lw;
}

void KittiMotsSample::load_all(){
    if(loaded) return;
    std::unique_lock<std::mutex> ul(mtx);
    if(loaded) return;
    {
        auto a=std::async(std::launch::async, [&](){
            left=safe_imread(paths["left"]);
            cv::cvtColor(left,lb,cv::COLOR_BGR2GRAY);
            lw=rgb3b2gray16(lb);

        });
        auto b=std::async(std::launch::async, [&](){
            right=safe_imread(paths["right"]);
            cv::cvtColor(right,rb,cv::COLOR_BGR2GRAY);
            rw=rgb3b2gray16(rb);
        });
        // on out of scope performs async
    }

    loaded=true;
}

}

