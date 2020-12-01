#pragma once
#include <mlib/utils/cvl/matrix_adapter.h>
#include <mutex>
#include <memory>

#include <opencv2/core.hpp>

#include <mlib/utils/mlibtime.h>
#include <mlib/utils/cvl/pose.h>
#include <mlib/cuda/devmemmanager.h>



namespace cvl{


class TemporalStereoStream{
public:

    void init(int rows, int cols, cvl::Matrix<float,3,3> K, float baseline);
    cv::Mat1f operator()(cv::Mat1f hostdisp0,cv::Mat1f hostdisp1, cvl::PoseD pose);


private:
    std::mutex mtx; // shared memory requires sync
    std::shared_ptr<DevMemManager> dmm=nullptr;
    std::shared_ptr<DevStreamPool> pool=nullptr;
    int disparities,cols,rows;
    // host disparity images
    cvl::MatrixAdapter<float> disp0,disp1;
    // intermediates:
    cvl::MatrixAdapter<float> disperror2,predicteddisp,disperror,dispout,depthtmp,fusedout, medianout, disp1medianfillin;
    cvl::MatrixAdapter<float> depth;

    cvl::Matrix<float,3,3> K;

    void show(cvl::MatrixAdapter<float> im, std::string name);

    double baseline;
    bool inited=false;
};

template<class T> cv::Mat_<T> toDisplayImage(cv::Mat_<T> dim){
    cv::Mat1f disp(dim.rows,dim.cols);
    for(int r=0;r<dim.rows;++r)
        for(int c=0;c<dim.cols;++c){
            float f=dim(r,c);
            if(f>128) f=128;
            if(f<0) f=0;

            disp(r,c)=f/128.0f;
        }
    return disp;
}
}// end namespace cvl
