#include <mlib/opencv_util/stereo.h>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <mlib/utils/random.h>
#include <bitset>
#include <iostream>
#include <mlib/utils/cvl/matrix.h>
#include <mlib/utils/cvl/matrix_adapter.h>
#include <mlib/utils/mlog/log.h>
#include <mutex>
#include <mlib/utils/mlibtime.h>
using std::cout;using std::endl;
namespace cvl{
constexpr int size=512;


namespace  {
std::vector<Vector4<int>> comps;

void set_comparisons()
{
    comps.reserve(size);
    // pattern size is 41x41? sure, do better later...


    for(int i=0;i<size;++i)
    {
        double span=10;

        if(i<size/2)            span=5;

        Vector2d p0(mlib::randu(-span,span), mlib::randu(-span,span));
        Vector2d p1=p0;
        while((p0-p1).norm()<span)
        {
            p1=Vector2d(mlib::randu(-span,span), mlib::randu(-span,span));
        }
        comps.push_back(Vector4<int>(p0[0], p0[1], p1[0], p1[1]));
        if(i<size/4)
        {
            // around center...
            comps.push_back(Vector4<int>(0, 0, p1[0], p1[1]));

        }
    }




    // lets try grid to grid instead.

}
std::mutex mtx;


}
std::vector<Vector4<int>>& comparisons()
{
    std::unique_lock ul(mtx);
    if(comps.size()==0)
        set_comparisons();
    return comps;
}


cv::Mat1f preproc(cv::Mat1b lg)
{
    cv::Mat1f lf(lg.rows, lg.cols);
    for(int r=0;r<lg.rows;++r)
        for(int c=0;c<lg.cols;++c)
            lf(r,c)= float((float(lg(r,c)) - 127.0)/256.0);
    return lf;
}

cv::Mat1b grey(cv::Mat1f pp)
{
    cv::Mat1b g(pp.rows,pp.cols);
    for(int r=0;r<pp.rows;++r)
        for(int c=0;c<pp.cols;++c)
            g(r,c)=uchar(pp(r,c)*256 +127);
    return g;
}
bool in(int row, int col, int rows, int cols)
{
    if(row<0) return false;
    if(col<0) return false;
    return row<rows && col <cols;
}

class BriefDescriptor
{
public:
    // really want std::bitset, but its not cuda compat...
    std::bitset<size> set;
    BriefDescriptor()=default;
    BriefDescriptor(const cv::Mat1f& lp, int row, int col)
    {
        auto& cmps=comparisons();
        int rows=lp.rows;
        int cols=lp.cols;
        for(uint i=0;i<set.size();++i)
        {
            auto v=cmps[i];
            //cout<<v<<endl;
            int r0=row+v[0];
            int c0=col+v[1];
            int r1=row+v[2];
            int c1=col+v[3];

            if((in(r0,c0,rows, cols) && in(r1,c1,rows,cols))) {
                //cout<<"lp(r0,c0)<(lp(r1,c1): "<<lp(r0,c0)<<"<"<<lp(r1,c1)<<"-"<<margin<<endl;
                // use margin for base, but not target, would give robustness...
                set[i] = (lp(r0,c0)<lp(r1,c1));
            }
            else
                set[i]=true;
        }
    }
    inline int match(const BriefDescriptor& b) const
    {
        return (set^b.set).count();
    }

};
std::string str(const BriefDescriptor& b){
    std::stringstream ss;
    ss<<b.set;
    //for(int i=0;i<b.set.size();++i)
    //ss<<b.set[i]<<endl;
    return ss.str();
}

std::shared_ptr<MAWrapper<BriefDescriptor>> compute(cv::Mat1f lp)
{
    std::shared_ptr<MAWrapper<BriefDescriptor>>
            descs=create_matrix<BriefDescriptor>(lp.rows,lp.cols);
    for(int r=0;r<lp.rows;++r)
        for(int c=0;c<lp.cols;++c)
            descs->ma(r,c)=BriefDescriptor(lp,r,c);
    return descs;
}

cv::Mat1b display_disparity(cv::Mat1f disparity)
{
    cv::Mat1b ret(disparity.rows,disparity.cols);
    for(int r=0;r<disparity.rows;++r)
        for(int c=0;c<disparity.cols;++c)
        {
            float d=disparity(r,c);
            if(d>255) d=0;
            if(d<0) d=0;


            ret(r,c)=4*d;
        }
    return ret;
}


cv::Mat1f stereo(cv::Mat3b l, cv::Mat3b r,int max_disparity)
{
    cv::Mat1b lg,rg;
    cv::cvtColor(l,lg,cv::COLOR_BGR2GRAY);
    cv::cvtColor(r,rg,cv::COLOR_BGR2GRAY);
    return stereo(lg,rg,max_disparity);
}
cv::Mat1f stereo(cv::Mat1b l,
                 cv::Mat1b r,
                 int max_disparity)
{
    cv::Mat1f lg,rg;
    cv::Mat1f lfs=preproc(l);
    cv::Mat1f rfs=preproc(r);
    return stereo(lfs,rfs,max_disparity);
}
namespace  {


mlib::NamedTimerPack ntp;
auto& desc_timer=ntp["descriptor"];
auto& match_timer=ntp["match"];
}
cv::Mat1f stereo(cv::Mat1f lfs, cv::Mat1f rfs, int max_disparity)
{
    lfs=lfs.clone();
    rfs=rfs.clone();

    // step one, transform into the right domain.
    // I have three bytes, so simplest is a float descriptor of greyscale.
    // but extracting some kind of descriptor would be better.
    // lets start with simplicity.

    cv::blur(lfs,lfs,cv::Size(3,3));
    cv::blur(rfs,rfs,cv::Size(3,3));
    cv::blur(lfs,lfs,cv::Size(3,3));
    cv::blur(rfs,rfs,cv::Size(3,3));


    // so now I have my thing, lets compute the descriptors
    //desc_timer.tic();
    std::shared_ptr<MAWrapper<BriefDescriptor>> lbd = compute(lfs);
    std::shared_ptr<MAWrapper<BriefDescriptor>> rbd = compute(rfs);
    //desc_timer.toc();
    int rows=lbd->ma.rows;
    int cols=lbd->ma.cols;
    cv::Mat1f disparityLR(rows,cols,0.0f);

    // now match along epi line looking for L in R



    //match_timer.tic();
    for(int r=21;r<rows-21;++r)
    {
        for(int c=21;c<cols-21;++c)
        {
            int bestv=size;
            int best_index=0;
            int second_best=size;
            BriefDescriptor query=lbd->ma(r,c);


            for(int d=-1;d< max_disparity;++d)
            {
                if(!(c-d-21>0)) continue;

                int v=query.match(rbd->ma(r,c-d));
                if(v<bestv)
                {

                    bestv=v;
                    best_index=d;
                    if(d==-1) best_index=0;
                }else {
                    if(v<second_best)
                    {
                        second_best=v;
                    }

                }


            }
              if(bestv<size*0.3)
          //  if(bestv<size*0.1 && bestv+size*0.1 <second_best)
                disparityLR(r,c) = best_index;
        }
    }
   // match_timer.toc();
   // cout<<ntp<<endl;
    return disparityLR;

    cv::imshow("disparity LR",display_disparity(disparityLR));

    cv::Mat1f disparityRL(rows,cols,-1.0f);
    for(int r=25;r<rows-25;++r){

        for(int c=25;c<cols-25;++c){
            int bestv=size;
            int best_index=0;
            BriefDescriptor query=rbd->ma(r,c);

            for(int d=-1;d<max_disparity;++d)
            {
                if(!(c+d+25>0)) continue;

                int v=query.match(lbd->ma(r,c+d));
                if(v<bestv)
                {
                    bestv=v;
                    best_index=d;
                    if(d==-1) best_index=0;
                }
            }
            if(bestv<size*0.5)
                disparityRL(r,c) = best_index;
        }
    }

    cv::imshow("disparity RL",display_disparity(disparityRL));


    cv::Mat1f disparity(rows,cols,-1.0f);
    for(int r=0;r<rows;++r){
        for(int c=0;c<cols;++c){
            float dlr=disparityLR(r,c);
            float drl=disparityRL(r,c);
            if(std::abs(dlr-drl)<2)
                disparity(r,c) = dlr;

        }
    }
    return disparity;
}

cv::Mat3b offset_left(cv::Mat3b rgb, int cols)
{
    cv::Mat3b ret(rgb.rows, rgb.cols,cv::Vec3b(0,0,0));
    for(int r=0;r<rgb.rows;++r)
        for(int c=0;c<rgb.cols-cols;++c)
            ret(r,c+cols)=rgb(r,c);
    return ret;
}

std::vector<double>
sparse_stereo(cv::Mat1f l, cv::Mat1f r, int max_disparity, std::vector<cvl::Vector2d> ys){
    std::vector<Vector2f> tmps;tmps.reserve(ys.size());
    for(const auto& y:ys)
        tmps.push_back(y);
    return sparse_stereo(l,r,max_disparity,tmps);
}
std::vector<double>
sparse_stereo(cv::Mat1f l,
              cv::Mat1f r,
              int max_disparity,
              std::vector<cvl::Vector2f> ys)
{

    std::vector<double> disparities;disparities.resize(ys.size(), -1);


    // step one, transform into the right domain.
    // I have three bytes, so simplest is a float descriptor of greyscale.
    // but extracting some kind of descriptor would be better.
    // lets start with simplicity.
    cv::Mat1f lfs, rfs;
    cv::blur(l,lfs,cv::Size(3,3));
    cv::blur(r,rfs,cv::Size(3,3));

    // so now I have my thing, lets compute the descriptors


    match_timer.tic();
    for(int index=0;index<ys.size();++index)
    {


        int row=ys[index][0];
        int col=ys[index][1];
        if(row<0||col<0) continue;
        BriefDescriptor query(lfs,row,col);

        int bestv=size;
        int best_index=0;


        for(int d=-1;d< max_disparity;++d)
        {
            BriefDescriptor target(rfs,row, col-d);
            int v=query.match(target);
            if(v<bestv)
            {

                bestv=v;
                best_index=d;
                if(d<-1) best_index=0;
            }
        }
        if(bestv<size*0.5)
            disparities[index]=best_index;
    }

    //match_timer.toc();
    //cout<<ntp<<endl;
    return disparities;
}

}
