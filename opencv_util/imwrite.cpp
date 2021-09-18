#include <mlib/opencv_util/imwrite.h>
#include <mlib/utils/files.h>
#include <opencv2/imgcodecs.hpp>
#include <mlib/utils/mlog/log.h>
#include <mlib/utils/mlibtime.h>
#include <mlib/utils/string_helpers.h>
namespace mlib{



bool write_image_safe(std::string pth, cv::Mat img) noexcept
{
    if(img.data==nullptr){
        mlog()<<"trying to write corrupt image"<<pth<<" size: "<<img.rows<<", "<<img.cols<<"\n";
        return false;
    }

    if(img.rows==0||img.cols==0){
        mlog()<<"trying to write empty image"<<pth<<"\n";
        return false;
    }


    mlib::create_or_throw(pth);

    auto p=fs::path(pth);
    std::string ext=p.extension();
    switch (img.type()) {
    case CV_8U:
        if(!(ext==".png"|| ext==".bmp"))
        {
            mlog()<<"warning writing image to unsupported format"<<pth<<"\n";
            break;
        }
    case CV_16U:
        if(ext==".png")
        {
            mlog()<<"warning writing image to unsupported format"<<pth<<"\n";
        }
        break;
    case CV_32F:
        if(ext!=".exr")
            mlog()<<"warning writing image to unsupported format"<<pth<<"\n";
        break;
    default:
        mlog()<<"missing format: "<<pth<<"\n";
    }
    std::string stem=p.stem();
    std::string path2=stem+".inprogress"+str(get_steady_now())+ext;
    if(ext.at(0)!='.') mlog()<<"extension error in write safe image!\n";

    bool res=cv::imwrite(path2,img);
    if(!res) return false;
    fs::rename(path2,pth);
    return fs::exists(pth);
}

std::future<bool> future_write_image(std::string path, cv::Mat img)noexcept{
    return std::async(std::launch::async, [path, img]()->bool{ return write_image_safe(path,img); });
}
std::map<std::string, bool> future_write_image(std::map<std::string, cv::Mat> images) noexcept{

    std::map<std::string, std::future<bool>> futures;
    for(auto [path,image]:images)
        futures[path]=future_write_image(path,image);
    std::map<std::string, bool> results;
    for(auto& [path,res]:futures)
        results[path]=res.get();
    return results;
}
bool all_good(const std::map<std::string, bool>& fwis)noexcept{
    for(const auto& [path,good]:fwis) if(!good) return false;
    return true;
}
}
