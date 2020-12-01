#pragma once
#include "mlib/cuda/cuda_helpers.h"
#include "mlib/cuda/result.h"



class Brief32 {
public:
    int desc[8];

    __device__  inline int distance(Brief32& b){
        int res=0;
        for(int i=0;i<8;++i)
            res+=__popc(desc[i] xor b.desc[i]);
        return res;
    }
};
class Brief64{
public:
    int desc[16];

    __device__  inline int distance(Brief64& b, int worst){
        int res=0;
        for(int i=0;i<16;++i){
            res+=__popc(desc[i] xor b.desc[i]);
            if(res>worst){
                return 512;
            }
        }
        return res;
    }
    __device__  inline int distance2(Brief64& b){
        int res=0;
        for(int i=0;i<16;++i){
            res+=__popc(desc[i] xor b.desc[i]);
        }
        return res;
    }
};

template<int Size> __global__ void match32(Brief32* a, int asize, Brief32* b, int bsize, Result<Size>* results){
    // single gx=asize,gy=1,gz=1
    // threads=32, read 32 descs, its tiny!



    int Aind=32*blockIdx.x + threadIdx.x;
    if(Aind>=asize)
        return;
    Result<Size> result;
    Brief32 A = a[Aind];

    for(int i=0;i<bsize;++i){// counting on the cashe! yeah seems to work just as well...
        result.insert(i,A.distance(b[i]));
    }
    results[Aind]=result;
    return;
}
template<int Size> __global__ void match64(Brief64* a, int asize, Brief64* b, int bsize, Result<Size>* results, int maxdist){
    // single gx=asize,gy=1,gz=1
    // threads=32, read 32 descs, its tiny!



    int Aind=32*blockIdx.x + threadIdx.x;


    Result<Size> result;
    Brief64 A;
    if(Aind<asize) // return here would lock at synchthreads...
        A = a[Aind];

    __shared__ Brief64 B;
    //__shared__ Brief64 B2;
    for(int i=0;i<bsize;++i){
        // counting on the cashe! yeah seems to work just as well...
        // not if I do early exits for the popc_, worth it? testing...
        //if(threadIdx.x==0)
        //  B=b[i];
        if(threadIdx.x<16)
            B.desc[threadIdx.x]=b[i].desc[threadIdx.x];

        __syncthreads();
        if(Aind<asize)
            result.insert(i,A.distance(B,maxdist));
    }

    results[Aind]=result;
    return;
}
template<int Size> __global__ void match642(Brief64* a, int asize, Brief64* b, int bsize, Result<Size>* results, int maxdist){
    // single gx=asize,gy=1,gz=1
    // threads=32, read 32 descs, its tiny!



    int Aind=32*blockIdx.x + threadIdx.x;


    Result<Size> result;
    Brief64 A;
    if(Aind<asize) // return here would lock at synchthreads...
        A = a[Aind];


    for(int i=0;i<bsize;++i){
        // counting on the cashe! yeah seems to work just as well...



        if(Aind<asize)
            result.insert(i,A.distance2(b[i]));
    }

    results[Aind]=result;
    return;
}
template<int Size> __global__ void match64q(Brief64* a, int asize, Brief64* b, int bsize, Result<Size>* results, int maxdist){
    // single gx=asize,gy=1,gz=1
    // threads=32, read 32 descs, its tiny!



    int Aind=32*blockIdx.x; // each block computes all for one of them

    Brief64 A=a[Aind];
    __shared__ int e[32];
    Result<Size> result;
    for(int i=0;i<bsize;i+=2){
    if(i+1<bsize){
        int B=b[i].desc[threadIdx.x];// works as long as the brief is a pure container and stored continously
        if(threadIdx.x<16)
            e[threadIdx.x]= __popc(A.desc[threadIdx.x] xor B);
        else
            e[threadIdx.x]= __popc(A.desc[threadIdx.x-16] xor B);
        __syncthreads();
        // oki I can probably sum them hierarchically but I doubt it makes a big difference...
        if(threadIdx.x==0){
            int res=0;
            for(int i=0;i<16;++i)
                res+=e[i];
            e[0]=res;
        }
        if(threadIdx.x==16){
            int res=0;
            for(int i=16;i<32;++i)
                res+=e[i];
            e[1]=res;
        }
        __syncthreads();
        if(threadIdx.x==0){
            result.insert(i,e[0]);
            result.insert(i+1,e[1]);
        }
        __syncthreads();
    }
    }
    if(threadIdx.x==0)
        results[Aind]=result;
    return;
}















