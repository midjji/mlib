#include <mlib/datasets/hilti/sample.h>
#include <mlib/opencv_util/convert.h>
#include <mlib/opencv_util/imshow.h>
#include <mlib/utils/string_helpers.h>

namespace cvl {
namespace hilti {
cv::Mat3b ImageSample::rgb(int i) const
{
    auto it=images.find(i);
    if(it==images.end())
    {
        mlog()<<"bad index: "<<i<<"\n";
        exit(1);
    }

    return image2rgb3b(it->second);

}
bool ImageSample::complete() const{return images.size()==5;}
cv::Mat1f ImageSample::grey1f(int i) const
{
    auto it=images.find(i);
    if(it==images.end())
    {
        mlog()<<"bad index: "<<i<<"\n";
        exit(1);
    }

    return image2grey1f(it->second, 1.0/255.0);

}


void ImageSample::show() const{
    mlog()<<"here\n";
    for(const auto& [id,image]:images) {
        mlog()<<"nhere\n";
        imshow(image,"hilti cam "+str(id));
        mlog()<<"nnhere\n";
    }
    mlog()<<"nnnhere\n";
}

}
}

