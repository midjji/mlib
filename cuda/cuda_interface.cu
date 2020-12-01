#include "mlib/cuda/cuda_interface.h"
#include "mlib/cuda/cuda_helpers.h"
#include "mlib/cuda/brief.cuh"


#include "mlib/sfm/opencv_util/cv.h"
#include <iostream>
#include "mlib/utils/random.h"

namespace cvl{
using std::cout;using std::endl;
using namespace mlib;

#define Size 4
// slow as hell first time... faster afterwards
std::vector<std::vector<cv::DMatch> > matchBrief(cv::Mat query, cv::Mat train,int maxdist){
   // cout<<"Start cuda brief matching"<<endl;


    assert(query.CONTINUOUS_FLAG);           assert(train.CONTINUOUS_FLAG);

    Result<Size>* resultHost=nullptr;


    if(query.cols==32){ // bytes!

        Result<Size>* result=cudaNew<Result<Size>>(query.rows);
        Brief32* Ahost=(Brief32*)query.data;
        Brief32* Bhost=(Brief32*)train.data;
        Brief32* A=nullptr;
        Brief32* B=nullptr;

        copy<Brief32>(Ahost,A,query.rows);
        copy<Brief32>(Bhost,B,train.rows);

        cudaDeviceSynchronize();
        dim3 grid((int)(31+query.rows)/32,1,1);
        dim3 threads(32,1,1);

        match32<Size><<<grid,threads>>>(A,query.rows,B,train.rows,result);
        cudaDeviceSynchronize();
        copy<Result<Size>>(result,resultHost,query.rows);
        cudaDeviceSynchronize();
        worked(cudaFree(A));
        worked(cudaFree(B));
        worked(cudaFree(result));
    }
    else{
        assert(query.cols==64);
        Result<Size>* result=cudaNew<Result<Size>>(query.rows);
        Brief64* Ahost=(Brief64*)query.data;
        Brief64* Bhost=(Brief64*)train.data;
        Brief64* A=nullptr;
        Brief64* B=nullptr;

        copy<Brief64>(Ahost,A,query.rows);
        copy<Brief64>(Bhost,B,train.rows);

        cudaDeviceSynchronize();
        dim3 grid((int)(31+query.rows)/32,1,1);
        dim3 threads(32,1,1);

        //match64<Size><<<grid,threads>>>(A,query.rows,B,train.rows,result, maxdist);
        match642<Size><<<grid,threads>>>(A,query.rows,B,train.rows,result, maxdist);
        //dim3 grid(query.rows,1,1);
        //dim3 threads(32,1,1);
        //match64q<Size><<<grid,threads>>>(A,query.rows,B,train.rows,result, maxdist);

        cudaDeviceSynchronize();
        copy<Result<Size>>(result,resultHost,query.rows);
        cudaDeviceSynchronize();
        worked(cudaFree(A));
        worked(cudaFree(B));
        worked(cudaFree(result));
    }
    cudaDeviceSynchronize();

    std::vector<std::vector<cv::DMatch>> matches;matches.reserve(query.rows);
    for(int i=0;i<query.rows;++i){
        std::vector<cv::DMatch> ms;ms.reserve(Size);
        for(int j=0;j<Size;++j){
            if(resultHost[i].error[j]<maxdist){
                if(!(resultHost[i].index[j]<train.rows))
                    std::cout<<resultHost[i].index[j]<<std::endl;
                assert(resultHost[i].index[j]<train.rows);

                ms.push_back(cv::DMatch(i, resultHost[i].index[j],resultHost[i].error[j]));
            }
        }
        //if(ms.size()==1);
        matches.push_back(ms);
    }
   // cout<<"done cuda briefmatching"<<endl;

    delete resultHost;
    return matches;
}















}
