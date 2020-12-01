#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <mlib/kitti/mots/dataset.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <mlib/utils/random.h>
#include <bitset>
#include <mlib/utils/cvl/matrix_adapter.h>
#include <mlib/opencv_util/stereo.h>

using namespace cvl;
int any_in_radius(cv::Mat1b& bw, int row, int col, int kernel_half_size)
{
    // can be made much faster... dont care
    for(int r=std::max(int(0),row - kernel_half_size -1);
        r<std::min(bw.rows,row+kernel_half_size);++r)
        for(int c=std::max(int(0),col - kernel_half_size -1);
            c<std::min(bw.cols, col+kernel_half_size);++c)
            if(bw(r,c)>0) return 1;
    return 0;
}

cv::Mat1b dilate(cv::Mat1b im,
                 int steps,
                 int kernel_half_size=1)
{
    // logical or over kernel per position steps times
    cv::Mat1b out(im.rows,im.cols);
    out=im.clone();
    for(int s=0;s<steps;++s)
        for(int r=0;r<im.rows;++r)
            for(int c=0;c<im.cols;++c)
                out(r,c) = any_in_radius(im,r,c,kernel_half_size);
    return out;
}





int main()
{
    cvl::KittiMotsDataset kmd("/storage/datasets/kitti/mots/");
    kmd.write_sample_paths("/home/mikael/tmp/mots.txt");
    auto s=kmd.get_sample(true,0,0);
    int n=0;
    while(s){
        cv::imshow("rgb0", s->rgb(0));
        cv::imshow("rgb1", s->rgb(1));
        cv::imshow("offset", offset_left(s->rgb(1),n++));
        cv::imshow("diff", s->rgb(0)- offset_left(s->rgb(1),n++));

        cv::Mat1b disparity = display_disparity(stereo(s->rgb(0),s->rgb(1),64));
        cv::imshow("disparity", disparity);

        uchar key=cv::waitKey(0);
        if(key=='n') s=kmd.next();
    }
    return 0;
}
