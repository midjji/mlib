#include <iostream>
#include <mlib/cuda/temporalstereo.h>
#include <mlib/utils/cvl/triangulate.h>
#include <mlib/cuda/common.cuh>
#include <mlib/cuda/median.cuh>
#include <mlib/cuda/opencv.h>
#include <mlib/vis/Visualization_helpers.h>
using std::cout;using std::endl;
using mlib::Timer;

namespace cvl{



    __global__ void triangulate(MatrixAdapter<float> disp,
                                MatrixAdapter<float> depth,
                                float f, float baseline){

        assert(disp.rows==depth.rows);
        assert(disp.cols==depth.cols);


        assert(disp.rows==gridDim.x);
        int row=blockIdx.x;
        assert(gridDim.y*blockDim.x>=disp.cols);
        int col=blockIdx.y*blockDim.x+threadIdx.x;


        if(row<50) return;
        if(row>disp.cols-50) return;
        if(col<50) return;
        if(col>disp.cols-50) return;

        if(col>disp.cols) return;
        float dispf=disp(row,col);
        if(dispf<0) return;
        if(dispf<6) dispf=6;
        float depthf=triangulate<float>(f,baseline,dispf);
        depth(row,col)=depthf;
    }
    __global__ void predictdisparity(MatrixAdapter<float> disparity,
                                     MatrixAdapter<float> disparityprediction,
                                     Matrix<float,3,3> K,
                                     Matrix<float,3,3> R,
                                     Vector3<float> t, float baseline){

        assert(disparity.rows == disparityprediction.rows);
        assert(disparity.cols == disparityprediction.cols);
        assert(disparity.rows == gridDim.x); // kernel req one per row
        int row=blockIdx.x;
        int col=blockIdx.y*blockDim.x+threadIdx.x;
        if(row<350) return;
        if(row>disparity.cols-150) return;
        if(col<150) return;
        if(col>disparity.cols-150) return;


        float dispf=disparity(row,col);
        if(dispf<0) return; // leaves the value at -1
        if(dispf<0.01) dispf=0.01; // keeps the values resonably small
        // triangulate the depth in the Left camera
        float depthf=triangulate<float>(K(0,0),baseline,dispf);
        // what point is beeing measured
        Vector3<float> y(col,row,1.0f);
        // transform it to pinhole normalized coordinates
        Vector3<float> yn=K.inverse()*y;
        // compute the 3D point
        Vector3<float> x=yn*depthf;

        // transform it to the new camera

        x=R*x +t;

        float outdepth=x[2];

        if(outdepth<=0) return;
        x=K*x;
        Vector2<float> yr=x.dehom();

        if(yr[0]<50) return;
        if(yr[1]<50) return;
        if(yr[0]>=disparityprediction.cols-50) return;
        if(yr[1]>=disparityprediction.rows-50) return;
        float dispout=(K(0,0)*baseline)/depthf;
        // return it to disparity=>
        disparityprediction(round(yr[1]),round(yr[0]))=dispout;

    }

    template<class T>
    __global__ void medianfillin(MatrixAdapter<T> in, MatrixAdapter<T> out)
    {

        assert(in.rows == out.rows);
        assert(in.cols == out.cols);
        assert(in.rows == gridDim.x); // kernel req one per row
        int row=blockIdx.x; // one per row
        int col=blockIdx.y*blockDim.x +threadIdx.x; // one per (col +31)/32
        if(!(col<in.cols)) return;

        float loc[9];
        if (row > 0 && col > 0 && row < in.rows - 1 && col < in.cols - 1) {
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    loc[r * 3 + c] = in(row + r - 1, col + c - 1);


            if (loc[4] < 0)
                out(row, col) = median(loc, 9);
        }
        else
            out(row,col) =in(row,col); // dont change stuff on the edges
    }

    __global__ void getError(MatrixAdapter<float> disparity,
                             MatrixAdapter<float> prediction,
                             MatrixAdapter<float> error){

        // check size
        assert(disparity.rows==prediction.rows);
        assert(disparity.cols==prediction.cols);
        assert(error.cols==prediction.cols);
        assert(error.rows==prediction.rows);
        // check configuration
        assert(gridDim.x==disparity.rows);
        assert(gridDim.y*blockDim.x>=disparity.cols);

        int row=blockIdx.x;
        int col=blockIdx.y*blockDim.x+threadIdx.x;
        if(!(col<disparity.cols)) return;
        float e=0;
        float disp=disparity(row,col);
        float pred=prediction(row,col);
        if(disp>3)
        if(pred>3){

            float norm=disp;if(pred>disp) norm =pred;

            e=fabs(disp-pred)*128/(norm+1.0f);
        }
        if(e>128) e=128;


        error(row,col)=e;
    }
    __global__ void fuse(MatrixAdapter<float> disparity,
                         MatrixAdapter<float> prediction,
                         MatrixAdapter<float> fused){

        // check size
        assert(disparity.rows==prediction.rows);
        assert(disparity.cols==prediction.cols);
        assert(fused.cols==prediction.cols);
        assert(fused.rows==prediction.rows);
        // check configuration
        assert(gridDim.x==disparity.rows);
        assert(gridDim.y*blockDim.x>=disparity.cols);

        int row=blockIdx.x;
        int col=blockIdx.y*blockDim.x+threadIdx.x;
        if(!(col<disparity.cols)) return;

        float disp=disparity(row,col);
        float pred=prediction(row,col);
        float fuse=-1;
        // missing data cases
        if(disp<0 && pred >0)
            fuse=pred;
        if(disp>0 && pred<0)
            fuse=disp;

        if(disp>0 && pred>0){
            // plain weighted average
            // if the disparity is within a threshold fuse them?
            if(fabs(disp-pred)<2)
                fuse=(disp*2.0f+pred)/3.0f;
            else
                // just the latest value,
                fuse=disp;
        }

        // the further away it is the better the fusion should be and inversly
        // alternatives
        // imos?

        fused(row,col)=fuse;
    }

    void TemporalStereoStream::show(MatrixAdapter<float> im, std::string name){
        cv::Mat1f im0=download2Mat(dmm,im);
        im0=toDisplayImage(im0);
        vis::showImage(name,im0);
    }


    void TemporalStereoStream::init(int rows, int cols, cvl::Matrix<float,3,3> K, float baseline){
        if(inited) return;
        std::unique_lock<std::mutex> ul(mtx);// the images are allocated to new memory
        cudaFree(0); //shoud ensure the cuda context is created...
        dmm=std::make_shared<DevMemManager>();
        pool=std::make_shared<DevStreamPool>(1);
        // host disparity images
        disp0=dmm->allocate<float>(rows,cols);
        disp1=dmm->allocate<float>(rows,cols);
        // host depth image, just for testing
        depth=dmm->allocate<float>(rows,cols);
        disperror=dmm->allocate<float>(rows,cols);
        disperror2=dmm->allocate<float>(rows,cols);
        fusedout=dmm->allocate<float>(rows,cols);
        predicteddisp=dmm->allocate<float>(rows,cols);
        disp1medianfillin=dmm->allocate<float>(rows,cols);
        depthtmp=dmm->allocate<float>(rows,cols);
        medianout=dmm->allocate<float>(rows,cols);
        this->rows=rows;
        this->cols=cols;
        this->K=K;
        this->baseline=baseline;
        inited=true;
    }

    cv::Mat1f TemporalStereoStream::operator()(cv::Mat1f hostdisp0,
                                               cv::Mat1f hostdisp1, cvl::PoseD pose){
        std::unique_lock<std::mutex> ul(mtx);// the images are allocated to new memory

        cv::Mat1f refdisp;
        if(!inited) return refdisp;

        // upload

        dmm->upload(convertFMat(hostdisp0),disp0);
        dmm->upload(convertFMat(hostdisp1),disp1);

        setAllDev(predicteddisp,pool->stream(0),-1.0f);
        setAllDev(disp1medianfillin,pool->stream(0),-1.0f);


        // triangulate
        dim3 grid(rows,(cols+31)/32,1);
        dim3 threads(32,1,1);
        triangulate<<<grid,threads,0,pool->stream(0)>>>(disp1,depth,K(0,0),fabs(baseline));
        // fill holes
        medianfillin<<<grid,threads,0,pool->stream(0)>>>(disp1,disp1medianfillin);


        // predict the disparity
        Matrix3x3<float> R=pose.getR();
        Vector3<float> t=pose.getT();
        predictdisparity<<<grid,threads,0,pool->stream(0)>>>(disp0,predicteddisp,K,R,t,fabs(baseline));



        getError<<<grid,threads,0,pool->stream(0)>>>(disp1,predicteddisp,disperror);



        //medianfilter3x3<float,32><<<grid,threads,0,pool->streams[0]>>>(disperror,disperror2);
        fuse<<<grid,threads,0,pool->stream(0)>>>(disp1,predicteddisp,fusedout);

        pool->synchronize();

        /*
        // download  and view
        show(disp1medianfillin,"new disparity with median fillin");

        show(predicteddisp,"predicted disparity");
        //show(medianout,"median fillin refinement");
        show(disperror,"prediction error");
        //show(disperror2,"median error");
        show(fusedout,"fused result");
*/
        cv::Mat1f fused=download2Mat(dmm,fusedout);
        //return hostdisp1.clone();
        return fused;












    }
}

