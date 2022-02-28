
#include <opencv2/core.hpp>

#include <opencv2/highgui.hpp>


#include <mlib/datasets/stereo_datasets.h>
#include <mlib/utils/random.h>
#include <mlib/utils/cvl/matrix_adapter.h>
#include <mlib/opencv_util/stereo.h>
#include <unistd.h>

#include <mlib/utils/argparser.h>
#include <mlib/utils/string_helpers.h>

using std::cout;
using std::endl;
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


int main(int argc, char** argv)
{

    auto params = args(argc, argv, { {"dataset", "kitti"},
                                     {"max disparity", "60"},
                                     {"sequence", "0"},
                                     {"offset", "0"},
                       });
    if(params["tracker config"]=="default")
    {
        if(params["dataset"]=="daimler")
            params["tracker config"]="/home/"+std::string(getlogin())+"/co/stereovo/daimler_klt.dat";
        if(params["dataset"]=="kitti")
            params["tracker config"]="/home/"+std::string(getlogin())+"/co/stereovo/kitti_klt.dat";
        if(params["dataset"]=="hilti")
            params["tracker config"]="/home/"+std::string(getlogin())+"/co/stereovo/hilti_klt.dat";
    }



    int sequence=mlib::str2int(params["sequence"]);
    std::cout << "sequence: " << sequence<<"\n";
    int max_disparity=mlib::str2int(params["max disparity"]);
    std::cout << "max_disparity: " << max_disparity<<"\n";

    auto seq = buffered_daimler_sequence(950);
    if(params["dataset"]=="hilti")
        seq= buffered_hilti_sequence(sequence,0);
    if(params["dataset"]=="kitti")
        seq= buffered_kitti_sequence();
    mlib::Timer timer("stereo time");
    uchar key='x';
    while(true)
    {
        auto s=seq->next();

        if(!s) break;
        if(!s->has_stereo()) continue;

        cv::imshow("rgb0", s->rgb(0));
        cv::imshow("rgb1", s->rgb(1));
        // cv::imshow("offset", offset_left(s->rgb(1),n++));
        // cv::imshow("diff", s->rgb(0)- offset_left(s->rgb(1),n++));
        timer.tic();

        cv::Mat1f disparity = stereo(s->rgb(0),s->rgb(1),max_disparity);

timer.toc();
cout<<timer<<endl;

        cv::imshow("disparity", display_disparity(disparity));
        cv::imshow("sgm disparity", display_disparity(s->disparity_image()));
        cv::Mat1b dispdiff(disparity.rows,disparity.cols,0.0f);
        cv::Mat1b dispdiff3(disparity.rows,disparity.cols,0.0f);
        cv::Mat1f gtdisp=s->disparity_image();
        for(int r=0;r<disparity.rows;++r)
            for(int c=0;c<disparity.cols;++c)
            {
                float gtd=gtdisp(r,c);
                float d=disparity(r,c);
                if(d<0 || gtd<0 ) {dispdiff(r,c)=0;continue;}
                float diff=d - gtd;
                if(diff<0) diff=-diff;
                if(diff<0) diff=0;
                if(diff>255) diff=255;
                dispdiff(r,c)=diff;



                if(std::abs(diff)<10)
                    dispdiff3(r,c)=20*std::abs(diff);
                else
                    dispdiff3(r,c)=255;

            }
        cv::imshow("sgm disparity - est disparity", dispdiff);
        cv::imshow("less than 3 pixel wrong", dispdiff3);
        int c=838;int r=646;
        // gt is 30,
        cout<<"("<<r<< ", "<<c<<"): "<<30<< " "<< gtdisp(r,c)<<" "<<disparity(r,c)<<"\n";
        cv::waitKey(0);
    }


    return 0;
}

