#include <assert.h>
#include <string>
#include <iostream>
#include <cuda_runtime.h>
#include "cuda_helpers.h"
#include <sstream>

using std::cout;
using std::endl;

namespace cvl{
/**
 * @brief devicePointer
 * @param p the pointer
 * @return if the pointer is __null or allocated on the device, since __null can be either or
 * &p[5] works aleast for the length of the allocated data
 */
bool devicePointer(const void* p){

    cudaPointerAttributes attributes;
    cudaPointerGetAttributes(&attributes,p);
    if(p==nullptr)
        return false;
    return (attributes.type == cudaMemoryTypeDevice);
}
std::string getCudaErrorMsg(const cudaError_t& error){
    return cudaGetErrorString(error);
}

bool worked(const cudaError_t& error,std::string msg){
    if(error==cudaSuccess)
        return true;
    cout<<msg<<": "<<getCudaErrorMsg(error);
    return false;
}





std::string getDeviceData(){
    std::stringstream ss;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop,0);
    ss<<"\nDevice: "<<prop.name<<" "<<prop.totalGlobalMem/(1024*1024)<<"MB"<<" Compute: "<<prop.major<<"."<<prop.minor<<"\n";

    ss<<"Concurrent Kernels: "<<prop.concurrentKernels<<"\n";
    ss<<"L2 cache size:      "<<prop.l2CacheSize/(1024)<<"KB"<<"\n";
    ss<<"asyncEngineCount:   "<<prop.asyncEngineCount<<"\n";
    ss<<"Multiprocessors:    "<<prop.multiProcessorCount<<"\n";
    ss<<"Compute mode:       "<<prop.computeMode<<" 0 means shared"<<"\n";

    ss<<"Shared memory per block: "<<prop.sharedMemPerBlock/(1024)<<"KB\n";
    ss<<"Max threads per block:   "<<prop.maxThreadsPerBlock<<"\n";
    ss<<"Max Threads per dim:     "<<prop.maxThreadsDim<<"\n";
    ss<<"Max threads per multiprocessor: "<<prop.maxThreadsPerMultiProcessor<<"\n";
    return ss.str();
}


}
