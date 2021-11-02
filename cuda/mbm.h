#pragma once
#include <mutex>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <mlib/utils/mlibtime.h>
#include <mlib/utils/cvl/matrix_adapter.h>
#include <mlib/cuda/devmemmanager.h>
#include <mlib/

namespace cvl{

/**
 * @brief The MBMStereoStream class
 * Stream class for the MBM stereo example,
 * initialize, then use, thread safe, blocking,allows most image sizes
 *
 * This isnt a great stereo method, but it is implemented as written in the paper.
 * Its likely there are steps missing or heuristics which where excluded
 */
class MBMStereoStream{
public:

    void init(int disparities, int rows, int cols);
    cv::Mat1b operator()(cv::Mat1b Left,cv::Mat1b Right);
    cv::Mat1f disparity(cv::Mat1b Left,cv::Mat1b Right){
        cv::Mat1b d=(*this)(Left,Right);
        cv::Mat1f ds(d.rows,d.cols);
        for(int r=0;r<d.rows;++r)
            for(int c=0;c<d.cols;++c)
                ds(r,c)=d(r,c);
        return ds;
    }
    void displayTimers();
    mlib::Timer getTimer();
private:
    std::mutex mtx; // shared memory requires sync
    std::shared_ptr<DevMemManager> dmm=nullptr;
    std::shared_ptr<DevStreamPool> pool=nullptr;
    int disparities,cols,rows;
    cvl::MatrixAdapter<float> costs;
    cvl::MatrixAdapter<uchar> disps,disps2;
    cvl::MatrixAdapter<uchar> L0,R0;

    std::vector<cvl::MatrixAdapter<int>> adiffs;
    std::vector<cvl::MatrixAdapter<int>> sats;
    //cvl::MatrixAdapter<cvl::MatrixAdapter<int>> satsv;



    bool inited=false;
    bool inner_median_filter=false;
    bool showdebug=false;

    mlib::Timer timer=mlib::Timer("total");
    mlib::Timer mediantimer=mlib::Timer("median");
    mlib::Timer cumSumRowtimer=mlib::Timer("cum sum row");
    mlib::Timer cumSumColtimer=mlib::Timer("cum sum col");
    mlib::Timer adifftimer=mlib::Timer("adifftimer");

};

}// end namespace cvl
