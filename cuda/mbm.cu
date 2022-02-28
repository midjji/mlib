#include "mbm.h"
#include <iostream>
#include <mlib/cuda/cuda_helpers.h>
#include <opencv2/highgui.hpp>
#include <mlib/cuda/cuda_helpers.h>
#include <mlib/cuda/devmemmanager.h>
#include <mlib/opencv_util/cv.h>
#include <mlib/cuda/opencv.h>
#include <mlib/cuda/common.cuh>
#include <mlib/cuda/median.cuh>
#include <mlib/utils/memmanager.h>

using std::cout;using std::endl;
using mlib::Timer;
namespace cvl{






template<int ADIFFTHREADS>
__global__ void adiff(MatrixAdapter<uchar> L,
                      MatrixAdapter<uchar> R,
                      MatrixAdapter<int>  res){


    assert(L.rows==R.rows);
    assert(L.cols==R.cols);
    assert(res.cols==L.cols);
    assert(res.rows==L.rows);

    int row=blockIdx.x;
    int col=blockIdx.y*ADIFFTHREADS +threadIdx.x;
    if(!(col<L.cols)) return;
    if(!(row<L.rows)) return;

    int Lv=L(row,col);
    int Rv=R(row,col);
    int tmp=sqrtf((Lv-Rv)*(Lv-Rv));
    //if(tmp<0)tmp=-tmp;
    //if(tmp>255) tmp=255;
    res(row,col)=tmp;
}






template<int CUMSUMCOLTHREADS>
__global__

/**
         * @brief cumSumCol
         * @param adiff
         * @param out
         * requires one block per row and 32 threads...
         */
void cumSumCol(MatrixAdapter<int> adiff,
               MatrixAdapter<int>  out)
{
    assert(adiff.rows==out.rows);
    assert(adiff.cols==out.cols);
    assert(CUMSUMCOLTHREADS==blockDim.x);




    // VERSION 1
    //read 32 using joint, then sum single,write 32 joint lower
    //mem and reg count... more than 2x as fasts as above in multiple sim kernel launches
    // beats using more shared mem..

    //__shared__ int acache[32]; // int here substantially improves performance because the conversions are made in parallel then!
    __shared__ int ocache[CUMSUMCOLTHREADS]; // cant use blockDim.x since static alloc is pref
    assert(gridDim.x==adiff.rows); // one per row
    int row=blockIdx.x;

    int cumsum=0;
    for(int col=0;col<adiff.cols;col+=CUMSUMCOLTHREADS){
        if(col+threadIdx.x<adiff.cols)
            ocache[threadIdx.x]=adiff(row,col+threadIdx.x);
        else
            ocache[threadIdx.x]=0;
        __syncthreads();


        if(threadIdx.x==0){
            // simplest solution is so very often right ...
            // atleast now that the system is so memory bound while running 64 at once
            for(int i=0;i<CUMSUMCOLTHREADS;++i){
                cumsum+=ocache[i];
                ocache[i]=cumsum;
            }
        }


        __syncthreads();
        if(col+threadIdx.x<adiff.cols)
            out(row,col+threadIdx.x)=ocache[threadIdx.x];
    }
}


#define CUMSUMROWTHREADS 64
__global__
void cumSumRow(MatrixAdapter<int>  csc)
{
    // VERSION 0
    // automatically becomes perfectly aligned
    int col=blockIdx.x*CUMSUMROWTHREADS+threadIdx.x;
    if(!(col<csc.cols))        return;
    int prev=0;
    for(int row=0;row<csc.rows;++row){
        prev+=csc(row,col);
        csc(row,col)=prev;
    }
}


template<unsigned int THREADS> __global__
void cumSumRowV2(MatrixAdapter<MatrixAdapter<int>> meta)
{

    MatrixAdapter<int>  csc=meta(blockIdx.y,0);
    __syncthreads();
    // VERSION 1
    // automatically becomes perfectly aligned
    int col=blockIdx.x*THREADS+threadIdx.x;

    if(csc.cols<=col)        return;
    int prev=0;
    for(int row=0;row<csc.rows;++row){
        prev+=csc(row,col);
        csc(row,col)=prev;
    }
}

__device__
inline uint getBlockErrorSum(MatrixAdapter<int>& sat, int row, int col, int halfwidthrow, int halfwidthcol){
    // check for bad input

    //
    // -1 due to the offset in the unpadded sat
    int startr=row-halfwidthrow -1;
    int endrow=row+halfwidthrow ;
    int startc=col-halfwidthcol -1;
    int endcol=col+halfwidthcol ;

    if(startc<0||startr<0) return 10000000;
    if(!(endrow<sat.rows)) return 10000000;
    if(!(endcol<sat.cols)) return 10000000;



    int A=sat(startr,startc);
    int B=sat(startr,endcol);
    int C=sat(endrow,startc);
    int D=sat(endrow,endcol);
    int err=D - B + A - C;
    return err;
}

__device__
inline uint getBlockErrorSum(MatrixAdapter<int>& sat, int row, int col, int halfwidth){
    return getBlockErrorSum(sat,row,col,halfwidth,halfwidth);
}


template<int UPDATEDISPARITYTHREADS>
__global__
/**
         * @brief updateDisparity
         * @param sat
         * @param disps
         * @param costs
         * one blockx per rows -32
         * one blocky per 32 cols
         * initializes costs and disps
         *
         */
void updateDisparity(MatrixAdapter<int> sat,
                     MatrixAdapter<uchar> disps,
                     MatrixAdapter<float> costs,
                     int disp){
#if 0
    if(threadZero()){
        printf("sat:%i,%i,disps:%i,%i,costs:%i,%i:\n",
               sat.rows, sat.cols, disps.rows, disps.cols, costs.rows, costs.cols);


        assert(sat.rows==disps.rows);
        assert(costs.rows==disps.rows);
        //assert(sat.cols==disps.cols);
        //assert(sat.cols==costs.cols);
    }
#endif
    int row=16+blockIdx.x; //one per row
    if(!(row<sat.rows-16))return;

    int col=blockIdx.y*UPDATEDISPARITYTHREADS + threadIdx.x;// 32 threads,blockIdx.y=(cols+31)/32

    if(row<10 ||row>sat.rows-10) return;

    uchar d=disp;
    //printf("%i",d);
    if(col>30 && col<sat.cols-30){

        float C7x7=getBlockErrorSum(sat,row,col,7,7);
        float C19x19=getBlockErrorSum(sat,row,col,9,9);

        float C61x1=1;
        if(row>31||row+31<sat.rows)
            C61x1=getBlockErrorSum(sat,row,col,30,1);

        float C1x61=1;
        if(col>31||col+31<sat.cols)
            C1x61=getBlockErrorSum(sat,row,col,1,30);

        float C61=C1x61;
        if(C61>C61x1)C61=C61x1;

        //float err=C7x7*C19x19*C61;
        float err=C7x7;

        float C0=costs(row,col);
        if((C0>err)||disp==0){
            costs(row,col)=err;
            disps(row,col)=d;
        }
    }
}


mlib::Timer MBMStereoStream::getTimer(){return timer;}
void MBMStereoStream::init(int disparities, int rows, int cols){

    if(inited) return;
    cudaFree(0); //shoud ensure the cuda context is created...
    dmm=std::make_shared<DevMemManager>();
    pool=std::make_shared<DevStreamPool>(disparities);
    std::unique_lock<std::mutex> ul(mtx);// the images are allocated to new memory
    //cout<<"init "<<endl;


    this->disparities=disparities;
    this->rows=rows;
    this->cols=cols;
    costs=dmm->allocate<float>(rows,cols);
    disps=dmm->allocate<uchar>(rows,cols);
    disps2=dmm->allocate<uchar>(rows,cols);

    //printdev(disps);
    adiffs.reserve(disparities);
    sats.reserve(disparities);
    for(int i=0;i<disparities;++i)        adiffs.push_back(dmm->allocate<int>(rows,cols-i));
    for(int i=0;i<disparities;++i)        sats.push_back(dmm->allocate<int>(rows,cols-i));

    L0=dmm->allocate<uchar>(rows,cols);
    R0=dmm->allocate<uchar>(rows,cols);
    MemManager mm;
    //MatrixAdapter<MatrixAdapter<int>> satsvhost=MatrixAdapter<MatrixAdapter<int>>::allocate(disparities,1);
    //mm.manage(satsvhost);

    //for(int i=0;i<disparities;++i) satsvhost(i,0)=sats[i];
    //satsv=dmm->upload(satsvhost);



    // cv::Mat1b disp;
    dmm->synchronize();
    inited=true;
    // cout<<"init done"<<endl;
}

cv::Mat1f toMat1f(cv::Mat1i sat){
    cv::Mat1f ret(sat.rows,sat.cols);
    double min,max;cv::Point min_loc,max_loc;
    cv::minMaxLoc(sat, &min, &max, &min_loc, &max_loc);
    for(int r=0;r<sat.rows;++r)
        for(int c=0;c<sat.cols;++c)
            ret(r,c)=((float)(sat(r,c)-min))/((float)(max-min));
    return ret;
}

void MBMStereoStream::displayTimers(){
    std::vector<mlib::Timer> ts={timer,mediantimer,cumSumRowtimer,cumSumColtimer,adifftimer};
    cout<<ts<<endl;
}
cv::Mat1b MBMStereoStream::operator()(cv::Mat1b Left, cv::Mat1b Right)
{
    std::unique_lock<std::mutex> ul(mtx);// the images are allocated to new memory
    if(!inited){        std::cerr<<"MBMStereoStream called before init!"<<endl;        exit(1);    }
    timer.tic();

    setAllDev(disps,pool->stream(666));


    MemManager mmi;
    if(true){
        auto l0=MatrixAdapter<uchar>::allocate(Left);mmi.manage(l0);
        auto r0=MatrixAdapter<uchar>::allocate(Right);mmi.manage(r0);

        dmm->upload(l0,L0);
        dmm->upload(r0,R0);
    }else{
            // leaks L0,R0!
        // does not work since something else then gets striding errors
        dmm->upload(convertFMat(Left),L0);
        dmm->upload(convertFMat(Right),R0);
    }
    dmm->synchronize();


    {
        // these kernels should be async => requries each in a stream of its own? YES!!!!!!
        adifftimer.tic();
        for(int i=0;i<disparities;++i){

            int offset=i;
            // owned by the L0,R0 matrixes
            MatrixAdapter<uchar> L=L0.getSubMatrix(0,offset,L0.rows,L0.cols-offset);
            MatrixAdapter<uchar> R=R0.getSubMatrix(0,0,R0.rows,R0.cols-offset);
            int thr=256;
            dim3 grid(L.rows,(L.cols+thr-1)/thr,1);
            dim3 threads(thr,1,1);
            adiff<256><<<grid,threads,0,pool->stream(i)>>>(L,R,adiffs[i]);
        }
        pool->synchronize(); // wait untill its needed! or for testing enable

        adifftimer.toc();

        //cout<<"adifftimer: "<<adifftimer<<endl;
    }

    //cout<<"variant 1 1"<<endl;
    {
        // problem, this solution kills the cache!, well less of a issue after I fixed some very basic pipelining, and alignment issues
        // not sure what I do wrong but this takes longer than opencv bm method, despite my integral image beeing quite alot faster

        {
            cumSumColtimer.tic();
            for(int i=0;i<disparities;++i){

                //int blocks=(adiffs[i].rows+31)/32;
                int blocks=adiffs[i].rows;
                dim3 grid(blocks,1,1);

                dim3 threads(32,1,1);
                cumSumCol<32><<<grid,threads,0,pool->stream(0)>>>(adiffs[i],sats[i]);

                if(showdebug){

                    pool->synchronize();
                    //printdev(sats[i]);

                    cv::Mat1f disp=toMat1f(download2Mat(dmm,sats[i]));
                    dmm->synchronize();

                    // print(tmp);
                    cv::imshow("colsum",disp);
                    cv::waitKey(0);
                }
            }
            pool->synchronize();// for speed testing


            cumSumColtimer.toc();
            //cout<<"cumSumColtimer: "<<cumSumColtimer<<endl;
        }




        {
            cumSumRowtimer.tic();
            for(int i=0;i<disparities;++i){
                int blocks=(CUMSUMROWTHREADS-1+sats[i].cols)/CUMSUMROWTHREADS;
                dim3 grid(blocks,1,1);
                dim3 threads(CUMSUMROWTHREADS,1,1);
                // streams are sequential!
                cumSumRow<<<grid,threads,0,pool->stream(0)>>>(sats[i]);// computes the row sums
                if(showdebug){
                    pool->synchronize();
                    cv::Mat1f disp=toMat1f(download2Mat(dmm,sats[i]));
                    dmm->synchronize();

                    // print(tmp);
                    cv::imshow("rowsum",disp);
                    cv::waitKey(0);
                }
            }

            pool->synchronize();
            cumSumRowtimer.toc();
            //cout<<"cumSumRowtimer: "<<cumSumRowtimer<<endl;
        }




    }

    //  cout<<"variant 1 3"<<endl;

    {
        Timer timer;timer.tic();
        for(int i=0;i<disparities;++i){
            constexpr int thr=32;
            int blockx=L0.rows;
            int blocky=(L0.cols+thr-1)/thr;
            dim3 grid(blockx,blocky,1);
            dim3 threads(thr,1,1);
            pool->synchronize(i);
            updateDisparity<32><<<grid,threads,0,pool->stream(i)>>>(sats[i],disps,costs,i);


            if(inner_median_filter && false){

                mediantimer.tic();


                dim3 blocks(disps.rows-1,(disps.cols+63)/64,1);
                dim3 threads(64,1,1);
                medianfilter3x3<uchar><<<blocks,threads,0,pool->stream(0)>>>(disps,disps2);

                pool->synchronize(0);
                std::swap(disps,disps2);

                mediantimer.toc();
            }



            if(showdebug){
                pool->synchronize();
                cv::Mat1b disp=download2Mat(dmm,disps);
                dmm->synchronize();
                // --- Grid and block sizes
                cv::imshow("disparities",disp);
                cv::waitKey(0);
            }
        }
        pool->synchronize(0);
        timer.toc();
        //cout<<"disparityTimer: "<<timer<<endl;
    }


    if(true){

        dim3 blocks(disps.rows,(disps.cols+63)/64,1);
        dim3 threads(64,1,1);
        medianfilter3x3<uchar><<<blocks,threads,0,pool->stream(0)>>>(disps,disps2);
        pool->synchronize(0);
        std::swap(disps,disps2);
    }

    // cant alloc with stride, but I can alloc a bigger one and return a submatrix...
    cv::Mat1b disp=download2Mat(dmm,disps);
    dmm->synchronize();
    timer.toc();
    //cout<<"median timer: "<<mediantimer<<endl;
    return disp;
}

}// end namespace cvl



