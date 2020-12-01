#pragma once
#include <mlib/utils/cvl/matrix_adapter.h>
#include "mlib/cuda/cuda_helpers.h"
template<int Size>
class Result{
public:
    mlib_host_device_ Result(){
       clear();
    }
    int index[Size];
    int error[Size];
    mlib_host_device_ void insert(int ind, int dist){
        // presorted list, most likely to be the worst match

        if(error[Size-1]>dist){
            index[Size-1]=ind;
            error[Size-1]=dist;
            sort();
        }
    }
    mlib_host_device_ void sort(){
        for(int i=Size-1;i>0;--i){
            if(error[i-1]>error[i]){
                swap(i-1,i);
            }else{
                break;
            }
        }
    }
    mlib_host_device_ inline void swap(int a, int b){
        int et=error[b];
        int it=index[b];
        error[b]=error[a];
        error[a]=et;
        index[b]=index[a];
        index[a]=it;
    }
    mlib_host_device_ void clear(){
        for(int i=0;i<Size;++i){
            error[i]=512;
            index[i]=-1;
        }
    }
    mlib_host_device_ inline int worst(){return error[Size-1];}
};
