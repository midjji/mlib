#pragma once
#include <mutex>
#include <memory>
#include <opencv2/core.hpp>
#include <complex>
#include <mlib/utils/cvl/tensor.h>
namespace cvl{


// doing getssd for all positions should be the same as mirroring the

template<class T,class U> void set(cv::Mat_<T>& im, U v){
    for(int i=0;i<im.rows*im.cols;++i)
            im(i)=T(v);
}
class BMStereo{
public:

    cv::Mat1b operator()(cv::Mat1b Left,cv::Mat1b Right){
        return compute(Left,Right,64);
    }
    cv::Mat1f compute(cv::Mat1f L, cv::Mat1f R, int max_disparity=255){

        std::shared_ptr<TensorWrapper<float,3>> cv=TensorWrapper<float,3>::allocate(L.rows,L.cols,max_disparity);
        auto costvolume=cv->tensor;
        cv::Mat1f cost_min(L.rows,L.cols);set(cost_min,255*11*11);
        cv::Mat1f disps(L.rows,L.cols);set(disps,-1);
        // compute diff
        int frm=7;
        int fcm=11;
        for(int r=0;r<L.rows;++r)
            for(int c=0;c<L.cols;++c){
                for(int d=0; d<max_disparity;++d){
                    // get cost at position: r,c vs r,c+d
                    double err=0;
                    for(int fr=-std::floor(frm/2);fr<std::ceil(frm/2.0);fr++)
                        for(int fc=-std::floor(fcm/2);fc<std::ceil(fcm/2.0);fc++){
                            if(r+fr<0 ||r+fr>=L.rows) {err+=1; continue;}
                            if(c+fc-d<0 ||c+fc-d>=L.cols) {err+=1; continue;}
                            if(c+fc<0 ||c+fc>=L.cols) {err+=1; continue;}
                            err+=std::abs(L(r+fr,c+fc) - R(r+fr,c+fc-d));
                        }
                    costvolume(r,c,d)=err;
                    if(cost_min(r,c)>err){
                        cost_min(r,c)=err;
                        disps(r,c)=d;
                    }
                }
            }
        return disps;
    }


};
/*
class BMNStereo{
public:

    cv::Mat1b operator()(cv::Mat1b Left,cv::Mat1b Right){
        return compute(Left,Right,64);
    }
    TensorAdapter<float,3> extract_features(cv::Mat1f im){

        TensorAdapter<float,3> im16=TensorAdapter<float,3>::allocate(L.rows,L.cols,16);
        for

        return im16;
    }


    cv::Mat1f compute(cv::Mat1f L, cv::Mat1f R, int max_disparity=255){

        //extract features
        TensorAdapter<float,3> L16=TensorAdapter<float,3>::allocate(L.rows,L.cols,16);
        TensorAdapter<float,3> R16=TensorAdapter<float,3>::allocate(L.rows,L.cols,16);


        TensorAdapter<float,3> costvolume=TensorAdapter<float,3>::allocate(L.rows,L.cols,max_disparity);
        cv::Mat1f cost_min(L.rows,L.cols);set(cost_min,255*11*11);
        cv::Mat1f disps(L.rows,L.cols);set(disps,-1);
        // compute diff
        int frm=7;
        int fcm=11;
        for(int r=0;r<L.rows;++r)
            for(int c=0;c<L.cols;++c){
                for(int d=0; d<max_disparity;++d){
                    // get cost at position: r,c vs r,c+d
                    double err=0;
                    for(int fr=-std::floor(frm/2);fr<std::ceil(frm/2.0);fr++)
                        for(int fc=-std::floor(fcm/2);fc<std::ceil(fcm/2.0);fc++){
                            if(r+fr<0 ||r+fr>=L.rows) {err+=1; continue;}
                            if(c+fc-d<0 ||c+fc-d>=L.cols) {err+=1; continue;}
                            if(c+fc<0 ||c+fc>=L.cols) {err+=1; continue;}
                            err+=std::abs(L(r+fr,c+fc) - R(r+fr,c+fc-d));
                        }
                    costvolume(r,c,d)=err;
                    if(cost_min(r,c)>err){
                        cost_min(r,c)=err;
                        disps(r,c)=d;
                    }
                }
            }
        return disps;
    }


};

*/











std::vector<std::complex<double>> compute_dft(std::vector<std::complex<double>> x){
    std::vector<std::complex<double>> dft;dft.reserve(x.size());
    double pi=3.14159265359;
    double N=x.size();
    for(uint k=0;k<x.size();++k)
        for(uint n=0;n<x.size();n++)
            dft[k] = x[n]*std::exp((k*n*2*pi/N)*std::complex<double>(0,1));
    return dft;
}






class MosseStereo{
public:

    cv::Mat1b operator()(cv::Mat1b Left,cv::Mat1b Right){
        return compute(Left,Right,64);
    }
    cv::Mat1f compute(cv::Mat1f L, cv::Mat1f R, int max_disparity=255){
        std::shared_ptr<TensorWrapper<float,3>> cv=TensorWrapper<float,3>::allocate(L.rows,L.cols,max_disparity);
        auto costvolume=cv->tensor;
        cv::Mat1f cost_min(L.rows,L.cols);set(cost_min,255*11*11);
        cv::Mat1f disps(L.rows,L.cols);set(disps,-1);
        // compute diff
        int frm=7;
        int fcm=11;
        for(int r=0;r<L.rows;r+=10)
            for(int c=0;c<L.cols;++c){

                // beräkna filtret innom patchen
                // beräkna svaret och skriv till







                for(int d=0; d<max_disparity;++d){
                    // get cost at position: r,c vs r,c+d
                    double err=0;
                    for(int fr=-std::floor(frm/2);fr<std::ceil(frm/2.0);fr++)
                        for(int fc=-std::floor(fcm/2);fc<std::ceil(fcm/2.0);fc++){
                            if(r+fr<0 ||r+fr>=L.rows) {err+=1; continue;}
                            if(c+fc-d<0 ||c+fc-d>=L.cols) {err+=1; continue;}
                            if(c+fc<0 ||c+fc>=L.cols) {err+=1; continue;}
                            err+=std::abs(L(r+fr,c+fc) - R(r+fr,c+fc-d));
                        }
                    costvolume(r,c,d)=err;
                    if(cost_min(r,c)>err){
                        cost_min(r,c)=err;
                        disps(r,c)=d;
                    }
                }
            }
        return disps;
    }


};













}// end namespace cvl
















