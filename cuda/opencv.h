#pragma once

#include <opencv2/core.hpp>

#include <memory>
#include <iostream>
#include <mlib/cuda/devmemmanager.h>
#include <mlib/utils/cvl/matrix_adapter.h>

namespace cvl{

#if 1
// cv::Mat owns the pointer
template<class T> MatrixAdapter<T> convertFMat(cv::Mat_<T> M){
    // I am dubious of this one!
assert(false && "does not always work!!!!");
    // could be offset!

    //return MatrixAdapter<T>((T*)M.data,M.rows,M.cols,M.step);
    return MatrixAdapter<T>((T*)(&M(0,0)),M.rows,M.cols,M.step);

}
#endif

template<class T>
/**
 * @brief download2Mat
 * @param dmm
 * @param disps
 * @return the cv::Mat which owns the host data
 *
 * \todo
 * - verify this really does take ownership
 * - replace using static method in DevMemManager, hard since I use its pool
 * - check its actually a device matrix
 * - dont alloc it using the dmm
 */
cv::Mat_<T> download2Mat(std::shared_ptr<DevMemManager> dmm,
                         MatrixAdapter<T>& im){
assert(im.getData()!=nullptr);

    MatrixAdapter<T> tmp=dmm->download(im);


    cv::Mat_<T> ret;
    ret=cv::Mat_<T>::zeros(im.rows,im.cols);

    for(uint r=0;r<im.rows;++r)
        for(uint c=0;c<im.cols;++c){

            ret(r,c)=tmp(r,c);
        }

    tmp.release();
    return ret;
    //cv::Mat_<T> big(disps.rows,disps.stride);
    //return big(cv::Rect(0,0,disps.cols,disps.rows));
}

template<class T> void print(cv::Mat_<T> img){
    for(int r = 0;r < img.rows;++r){
        std::cout<<"row: "<<r<<" - ";
        for(int c = 0;c < img.cols;++c)
            std::cout<<img(r,c)<<", ";
        std::cout<<"\n";
    }
}





}
