#include <thread>

#include <mlib/opencv_util/read_image.h>
#include <mlib/utils/files.h>
#include <opencv2/imgcodecs.hpp>
#include <mlib/utils/mlog/log.h>
#include <mlib/opencv_util/type2str.h>
#include <future>

namespace mlib{


template<class T> cv::Mat_<T> read_image_(std::string path)  noexcept{
    if(!fs::exists(path)) {
        mlog()<<"file not found: "<<path<<"\n";
        return cv::Mat_<T>(10,10);
    }

    cv::Mat img=cv::imread(path,cv::IMREAD_ANYDEPTH|cv::IMREAD_ANYDEPTH);
    if(img.rows==0|| img.cols==0|| img.data==nullptr){
        mlog()<<"failed to read image: "<<path<<"\n";
        return cv::Mat_<T>(10,10);
    }

    if(img.type()!=Tnum<T>())    {
        mlog()<<"found unexpected image format: "<<type2str(img.type())<<" "<<img.rows<<" "<<img.cols<<"\n";
        return cv::Mat_<T>(10,10);
    }
    return img;
}
template<class T> std::future<cv::Mat_<T>> future_read_image_(std::string path /* should not be ref, as async launch may mean ref has been invalidated!*/) noexcept
{
return std::async(std::launch::async, [path]()->cv::Mat_<T>{ read_image_<T>(path); });
}

// paralell read many
template<class T> std::map<int,cv::Mat_<T>> read_image_(const std::map<int,std::string>& paths) noexcept
{

    std::map<int,std::future<cv::Mat1b>> future_images;
    for(const auto& [id,path]:paths)
    {
        // bind path by copy
        future_images[id]=future_read_image_<T>(path);
    }
    std::map<int,cv::Mat_<T>> images;
    for(auto& [id,future_image]:future_images)
        images[id]=future_image.get();
    return images;
}



// more expressive errors than opencv
cv::Mat1b read_image1b(std::string path) noexcept{return read_image_<uint8_t>(path);}
cv::Mat1w read_image1w(std::string path) noexcept{return read_image_<uint16_t>(path);}
cv::Mat1f read_image1f(std::string path) noexcept{return read_image_<float>(path);}
cv::Mat3b read_image3b(std::string path) noexcept{return read_image_<cv::Vec3b>(path);}

// more expressive errors than opencv
std::future<cv::Mat1b> future_read_image1b(std::string path) noexcept{return future_read_image_<uint8_t>(path);}
std::future<cv::Mat1w> future_read_image1w(std::string path) noexcept{return future_read_image_<uint16_t>(path);}
std::future<cv::Mat1f> future_read_image1f(std::string path) noexcept{return future_read_image_<float>(path);}
std::future<cv::Mat3b> future_read_image3b(std::string path) noexcept{return future_read_image_<cv::Vec3b>(path);}

std::map<int,cv::Mat1b> read_image1b(std::map<int,std::string> paths) noexcept
{
    return read_image_<uint8_t>(paths);
}


}
