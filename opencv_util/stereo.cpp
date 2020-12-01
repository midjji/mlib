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

using std::cout;using std::endl;
namespace cvl{
constexpr int size=512;



// from to
std::vector<Vector4<int>> comparisons(){
    std::vector<Vector4<int>> comps;comps.reserve(size);
    // pattern size is 41x41? sure, do better later...


    for(int i=0;i<size;++i){
        int span=20;
        if(i<size*0.25)
            span=10;
        comps.push_back(Vector4<int>(mlib::randu<float>(-span,span),
                                     mlib::randu<float>(-span,span),
                                     mlib::randu<float>(-span,span),
                                     mlib::randu<float>(-span,span)));
    }

    // lets try grid to grid instead.




    return comps;
}

auto common_comparisons=comparisons();

cv::Mat1f preproc(cv::Mat1b lg){




    cv::Mat1f lf(lg.rows, lg.cols);
    for(int r=0;r<lg.rows;++r)
        for(int c=0;c<lg.cols;++c)
            lf(r,c)= float((float(lg(r,c)) - 127.0)/256.0);

    //return lf; // might not be necessary with margin
    // now lowpass filter it.
    cv::Mat1f lfs;
    cv::blur(lf,lfs,cv::Size(5,5));
    //cv::blur(lfs,lfs,cv::Size(5,5));
    return lfs;
}

cv::Mat1b grey(cv::Mat1f pp){
    cv::Mat1b g(pp.rows,pp.cols);
    for(int r=0;r<pp.rows;++r)
        for(int c=0;c<pp.cols;++c)
            g(r,c)=uchar(pp(r,c)*256 +127);
    return g;
}
bool in(int row, int col, int rows, int cols){
    if(row<0) return false;
    if(col<0) return false;
    return row<rows && col <cols;
}

class BriefDescriptor{
public:
    // really want std::bitset, but its not cuda compat...
    std::bitset<size> set;
    BriefDescriptor()=default;
    BriefDescriptor(cv::Mat1f& lp, int row, int col, float margin){
        int rows=lp.rows;
        int cols=lp.cols;
        for(uint i=0;i<set.size();++i){
            auto v=common_comparisons[i];
            //cout<<v<<endl;
            int r0=row+v[0];
            int c0=col+v[1];
            int r1=row+v[2];
            int c1=col+v[3];

            if((in(r0,c0,rows, cols) && in(r1,c1,rows,cols))){

                // cout<<"lp(r0,c0)<(lp(r1,c1): "<<lp(r0,c0)<<" "<<lp(r1,c1)<<endl;
                // use margin for base, but not target, would give robustness...
                set[i] = (lp(r0,c0)<(lp(r1,c1) -margin));
            }
            else
                set[i]=true;

        }


    }
    int match(BriefDescriptor b){
        return (set^b.set).count();
    }

};

std::shared_ptr<MAWrapper<BriefDescriptor>> compute(cv::Mat1f lp, float margin){
    std::shared_ptr<MAWrapper<BriefDescriptor>>
            descs=create_matrix<BriefDescriptor>(lp.rows,lp.cols);
    for(int r=0;r<lp.rows;++r)
        for(int c=0;c<lp.cols;++c)
            descs->ma(r,c)=BriefDescriptor(lp,r,c, margin);
    return descs;
}

cv::Mat1b display_disparity(cv::Mat1f disparity){
    cv::Mat1b ret(disparity.rows,disparity.cols);
    for(int r=0;r<disparity.rows;++r)
        for(int c=0;c<disparity.cols;++c)
            ret(r,c)=uint(disparity(r,c));
    return ret;
}


cv::Mat1f stereo(cv::Mat3b l, cv::Mat3b r,int max_disparity){
    cv::Mat1b lg,rg;
    cv::cvtColor(l,lg,cv::COLOR_BGR2GRAY);
    cv::cvtColor(r,rg,cv::COLOR_BGR2GRAY);
    return stereo(lg,rg,max_disparity);
}
cv::Mat1f stereo(cv::Mat1b l, cv::Mat1b r,int max_disparity){
    cv::Mat1f lg,rg;
    cv::Mat1f lfs=preproc(l);
    cv::Mat1f rfs=preproc(r);
    return stereo(lfs,rfs,max_disparity);
}
cv::Mat1f stereo(cv::Mat1f lfs, cv::Mat1f rfs, int max_disparity)
{
    // step one, transform into the right domain.
    // I have three bytes, so simplest is a float descriptor of greyscale.
    // but extracting some kind of descriptor would be better.
    // lets start with simplicity.


    // so now I have my thing, lets compute the descriptors
    std::shared_ptr<MAWrapper<BriefDescriptor>> lbd=compute(lfs,1.0/255.0);
    std::shared_ptr<MAWrapper<BriefDescriptor>> rbd=compute(rfs,0);
    int rows=lbd->ma.rows;
    int cols=lbd->ma.cols;
    cv::Mat1f disparityLR(rows,cols,0.0f);

    // now match along epi line looking for L in R

    for(int r=20;r<rows-20;++r){

        for(int c=0;c<cols;++c){
            int bestv=size;
            int best_index=0;
            BriefDescriptor query=lbd->ma(r,c);

            for(int d=0;d< max_disparity;++d){
                if(!(c-d-21>0)) continue;

                int v=query.match(rbd->ma(r,c-d));
                if(v<bestv)
                {

                    bestv=v;

                    best_index=d;
                }
            }
            if(bestv<size*0.3)
                disparityLR(r,c) = best_index;
        }        
    }
    return disparityLR;

    //cv::imshow("disparity LR",display_disparity(disparityLR));

    cv::Mat1f disparityRL(rows,cols,0.0f);
    for(int r=25;r<rows-25;++r){

        for(int c=25;c<cols-25;++c){
            int bestv=size;
            int best_index=0;
            BriefDescriptor query=rbd->ma(r,c);

            for(int d=0;d<64;++d){
                if(!(c+d+25>0)) continue;

                int v=query.match(lbd->ma(r,c+d));
                if(v<bestv)
                {
                    bestv=v;
                    best_index=d;
                }
            }
            if(bestv<size*0.3)
                disparityRL(r,c) = best_index;
        }
    }

    //cv::imshow("disparity RL",display_disparity(disparityRL));


    cv::Mat1f disparity(rows,cols,0.0f);
    for(int r=0;r<rows;++r){
        for(int c=0;c<cols;++c){
            float dlr=disparityLR(r,c);
            float drl=disparityRL(r,c);
            if(dlr-drl<2)
                disparity(r,c) = dlr;

        }
    }
    return disparity;
}

cv::Mat3b offset_left(cv::Mat3b rgb, int cols){
    cv::Mat3b ret(rgb.rows, rgb.cols,cv::Vec3b(0,0,0));
    for(int r=0;r<rgb.rows;++r)
        for(int c=0;c<rgb.cols-cols;++c)
            ret(r,c+cols)=rgb(r,c);
    return ret;
}
}
