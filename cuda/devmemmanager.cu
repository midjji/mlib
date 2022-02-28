#include "devmemmanager.h"
namespace cvl{
#if 1
DevMemManager::DevMemManager(){
    allocs.reserve(1024);
}
DevMemManager::~DevMemManager(){
    for(int i=0;i<allocs.size();++i){
        cudaFree(allocs[i]);allocs[i]=nullptr;
    }
}


void DevMemManager::synchronize(){
    pool.synchronize();
}
#endif
}// end namespace cvl
