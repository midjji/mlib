#pragma once
#include <cuda_runtime.h>
#include <vector>
namespace cvl{
/**
 * @brief The DevStreamPool class
 * Manages cuda streams in a convenient way
 */
class DevStreamPool{
public:
    DevStreamPool(int size=4);
    ~DevStreamPool();
    void synchronize();
    void synchronize(uint i);
    cudaStream_t& stream(uint i){
        return streams[i % streams.size()];
    }
private:
    std::vector<cudaStream_t> streams;
};
}
