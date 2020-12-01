#pragma once
#include <cuda_runtime.h>
#include <mlib/utils/cvl/MatrixAdapter.hpp>
#include <mlib/cuda/common.cuh>



namespace cvl{

    template<class T> __host__ __device__
    void split(T* data, T x, int& i, int& j){
        // assumes odd size vector
        // find a random value close to the median, mean is probably a good start
        while(i<=j){
            //scan from the left
            while(data[i]<x) ++i;
            //scan from the right
            while(x<data[j]) --j;

            //swap the two values
            if(i<=j){
                T tmp=data[i];
                data[i]=data[j];
                data[j]=tmp;
                i++;j--;
            }
        }
    }
    template<class T> __host__ __device__ T median(T* data, int size)
    {
        int L = 0;
        int R = size-1;
        int k = size / 2;
        int i;int j;
        while (L < R)
        {
            T x = data[k];
            i = L; j = R;
            split(data, x,i,j);
            if (j < k)  L = i;
            if (k < i)  R = j;
        }
        return data[k];
    }


template<class T>
__global__ void medianfilter3x3(MatrixAdapter<T> in, MatrixAdapter<T> out) {

    assert(in.rows == out.rows);
    assert(in.cols == out.cols);
    assert(in.rows == gridDim.x); // kernel req one per row
    /*
    printKernelConfiguration();
    if(threadZero()){
        printf("gridDim.y* blockDim.x>= in.cols %i == %i\n",gridDim.y*blockDim.x,in.cols);
    }
    */


    assert(gridDim.y * blockDim.x >= in.cols); // atleast one per col



    int row = blockIdx.x; // one block per row
    int col = blockIdx.y*blockDim.x +threadIdx.x; // one per col rounded up
    if (!(col < in.cols)) return; // blocks synch

    T loc[9];
    // several annoying special cases, for now leave them as they are

    if (row > 0 && col > 0 && row < in.rows - 1 && col < in.cols - 1) {
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                loc[r * 3 + c] = in(row + r - 1, col + c - 1);
        out(row , col ) = median(loc, 9);
    }
    else{
        // dont change the edges
        out(row,col)=in(row,col);
    }
}



    template<class T> void median(MatrixAdapter<T> in, MatrixAdapter<T> out, cudaStream_t& stream){
        dim3 blocks(in.rows,(in.cols+31)/32,1);
        dim3 threads(32,1,1);
        medianfilter3x3<<<blocks,threads,0,stream>>>(in,out);
    }












} // end namespace cvl
