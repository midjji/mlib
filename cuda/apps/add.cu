
/**
 * Derived from the nVIDIA CUDA 8.0 samples by
 *
 *   Eyal Rozenberg
 *
 * The derivation is specifically permitted in the nVIDIA CUDA Samples EULA
 * and the deriver is the owner of this code according to the EULA.
 *
 * Use this reasonably. If you want to discuss licensing formalities, please
 * contact the author.
 */


#include <iostream>
#include <memory>
#include <algorithm>
#include <cuda/runtime_api.hpp>
#include <cmath>
#include <kat/on_device/grid_info.cuh>
/*
namespace device {

template<class T> struct Memory
{
    T* data=nullptr;
    int size=0;
    Memory(int size) {
        data=allocate<T>(size);
    }
    ~Memory(){::cuda::memory::device::free(data);data=nullptr;size=0;}
    T* get(){return data;}

};
}

*/

template<class T> struct ArrayAdapter {
    T* data=nullptr;
    int size=0;
    ArrayAdapter()=default;
    ArrayAdapter(T* ptr, int size):data(ptr),size(size){}
    __host__ __device__ inline T& operator()(int i){return data[i];}
    __host__ __device__ inline const T& operator()(int i)  const{return data[i];}
    __host__ __device__ inline T& operator[](int i){return data[i];}
    __host__ __device__ inline const T& operator[](int i)  const{return data[i];}

};
__global__ void add(ArrayAdapter<float> A,
                    ArrayAdapter<float> B,
                    ArrayAdapter<float> C)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < A.size) { C[i] = A[i] + B[i]; }
}
/*
__global__ void add(const float *A, const float *B, float *C, int nume)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < nume) { C[i] = A[i] + B[i]; }
}
*/

/*
__host__ __device__ void add(dim3 gridDim,
                             uint3 blockIdx,
                             dim3 blockDim,
                             uint3 threadIdx,
                             int warpSize,
                             const float *A, const float *B, float *C, int nume)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < nume) { C[i] = A[i] + B[i]; }
}

template<class Kernel, class... Args>
void host_apply(Kernel kernel, dim3 blocks, dim3 threads, Args... args)
{


    for(uint bx=0;bx<blocks.x;++bx)
            for(uint by=0;by<blocks.y;++by)
                for(uint bz=0;bz<blocks.z;++bz)
                        for(uint tx=0;tx<threads.x;++tx)
                            for(uint ty=0;ty<threads.y;++ty)
                                for(uint tz=0;tz<threads.z;++tz) {
                                    uint3 blockIdx{bx,by,bz};
                                    uint3 threadIdx{tx,ty,tz};
                                    kernel(blocks, blockIdx, threads, threadIdx, args...);
                                }
}
*/







int main(void)
{

    if (cuda::device::count() == 0) {
        std::cerr << "No CUDA devices on this system" << "\n";
        exit(EXIT_FAILURE);
    }

    int nume = 50000;
    size_t size = nume * sizeof(float);
    std::cout << "[Vector addition of " << nume << " elements]\n";

    // If we could rely on C++14, we would  use std::make_unique
    std::vector<float> h_A(nume,0);
    std::vector<float> h_B(nume,0);
    std::vector<float> h_C(nume,0);

    auto generator = []() { return rand() / (float) RAND_MAX; };
    std::generate(h_A.begin(), h_A.begin() + nume, generator);
    std::generate(h_B.begin(), h_B.begin() + nume, generator);



    auto device = cuda::device::current::get();
    auto d_A = cuda::memory::device::make_unique<float[]>(device, nume);
    auto d_B = cuda::memory::device::make_unique<float[]>(device, nume);
    auto d_C = cuda::memory::device::make_unique<float[]>(device, nume);

    cuda::memory::copy(d_A.get(), &h_A[0], size);
    cuda::memory::copy(d_B.get(), &h_B[0], size);

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (nume + threadsPerBlock - 1) / threadsPerBlock;
    std::cout
        << "CUDA kernel launch with " << blocksPerGrid
        << " blocks of " << threadsPerBlock << " threads\n";

    // must
    ArrayAdapter a(d_A.get(),nume);
    ArrayAdapter b(d_B.get(),nume);
    ArrayAdapter c(d_C.get(),nume);
    cuda::launch(add,        cuda::launch_configuration_t( blocksPerGrid, threadsPerBlock ),a,b,c   );

    cuda::memory::copy(&h_C[0], d_C.get(), size);

    // Verify that the result vector is correct
    for (int i = 0; i < nume; ++i) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)  {
            std::cerr << "Result verification failed at element " << i << "\n";
            exit(EXIT_FAILURE);
        }
    }

    std::cout << "Test PASSED\n";
    std::cout << "SUCCESS\n";
    return 0;
}
