#pragma once


/** Common cuda stuff needs to be forcibly inlined or templated
 * to work due to a bug/lack of feature in cuda parsing */

namespace cvl{








    __forceinline__ __device__
    // number of threads
     uint getThreads(){
        return blockDim.x*blockDim.y*blockDim.z;
    }
    __forceinline__ __device__
    // number of blocks
     uint getBlocks(){
        return gridDim.x*gridDim.y*gridDim.z;
    }

    __forceinline__ __device__
/// return the unique thread which always exists, for printing purposes
     bool threadZero(){
        return ((blockIdx.x==0) &&
                (blockIdx.y==0) &&
                (blockIdx.z==0)&&
                (threadIdx.x==0) &&
                (threadIdx.y==0) &&
                (threadIdx.z==0) );
    }
    /// simple debugging functions, shows the kernel configuration
    __forceinline__ __device__  void printKernelConfiguration(){
        if(threadZero()){
            printf("threads:    %i\n",getThreads());
            printf("blocks:     %i\n",getBlocks());
            printf("blockDim.x: %i\n",blockDim.x);
            printf("blockDim.y: %i\n",blockDim.y);
            printf("blockDim.z: %i\n",blockDim.z);
            printf("gridDim.x:  %i\n",gridDim.x);
            printf("gridDim.y:  %i\n",gridDim.y);
            printf("gridDim.z:  %i\n",gridDim.z);
        }
    }


    template<class T>
    __global__ void printKernel(cvl::MatrixAdapter<T> m) {
        if(threadZero()) {
            for (int r = 0; r < m.rows; ++r) {
                printf("row: %i - ", r);
                for (int c = 0; c < m.cols; ++c){
                    float val=m(r,c);printf("%f, ", val);
                }

                printf("\n", r);
            }
            printf("Rows: %i, Cols: %i",m.rows,m.cols);
        }
        __syncthreads();
    }

    template<class T>
    void printdev(cvl::MatrixAdapter<T>  m){
        cudaDeviceSynchronize();
        dim3 grid(1,1,1);
        dim3 threads(1,1,1);
        printKernel<<<grid,threads>>>(m);
        cudaDeviceSynchronize();
    }




    template<class T>
    __global__ void setAllK(MatrixAdapter<T> m, T val){
        int row=blockIdx.x;
        int col=blockIdx.y*32+threadIdx.x;
        if(row<m.rows)
        if(col<m.cols)
            m(row,col)=val;
    }
    template<class T>
    void setAllDev(MatrixAdapter<T> m, cudaStream_t& stream, T val=0){
        dim3 grid(m.rows,(m.cols+31)/32,1);
        dim3 threads(32,1,1);
        setAllK<<<grid,threads,0,stream>>>(m,val);
    }
    template<class T>
    void setAllDev(MatrixAdapter<T> m, T val){
        dim3 grid(m.rows,(m.cols+31)/32,1);
        dim3 threads(32,1,1);
        setAllK<<<grid,threads,0>>>(m,val);
    }
} // end namespace cvl
