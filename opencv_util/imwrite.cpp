#include <mlib/opencv_util/imwrite.h>
#include <mlib/utils/files.h>
#include <opencv2/imgcodecs.hpp>
#include <mlib/utils/mlog/log.h>
namespace mlib{
bool write_image_safe(std::string pth, cv::Mat img){
    mlib::create_or_throw(pth);
    auto p=fs::path(pth);
    std::string ext=p.extension();
    std::string stem=p.stem();
    std::string path2=stem+".inprogress"+ext;
    if(ext.at(0)!='.') mlog()<<"extension error in write safe image!\n";
    bool res=cv::imwrite(path2,img);
    fs::rename(path2,pth);
    return res;
}
}
