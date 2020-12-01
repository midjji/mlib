#pragma once
#include <cuda_runtime.h>
#include <mlib/cuda/cuda_helpers.h>
#include <mlib/utils/cvl/matrix_adapter.h>


#include <mutex>
#include <mlib/cuda/devstreampool.h>
#include <iostream>
#include <memory>
//#include <mlib/utils/informative_asserts.h>

namespace cvl{

namespace cuda {

///////////////////////// old stuff /////////////////////////////////////

template<class T>
/// calls release on T in destructor
class MemWrapper
{
public:
    T t;
    MemWrapper(T t):t(t){}
    /// Note that the explicit difference is to ensure that the smart pointer needs not have a trivial constructor
    ~MemWrapper(){t.release();}
    // remove copy and assign?    
};
}

/**
 * @brief The DevMemManager class
 * convenience class for dev memory management...
 * Keeps track of size? well not really... hmm
 * synchronize up and downloads, remember that alloc is synced!, ideally prealloc using allocate.
 * allocate takes care of stride alignment automatically, but returned matrixes wont nec be continuous
 * \todo keep track of sizes better
 */
class DevMemManager{
public:

    DevMemManager();
    ~DevMemManager();

    template<class T>
    /**
     * @brief allocate on device
     * @param rows
     * @param cols
     * @return a automatically strided matrix! Always use these on device!
     */
    MatrixAdapter<T> allocate(uint rows, uint cols){
        uint stride=MatrixAdapter<T>::getStride(cols);
        assert(stride>=cols);
        T* data=(T*)cudaNew<char>(rows*stride);
        allocs.push_back((void*)data);
        return MatrixAdapter<T>(data,rows,cols,stride);
    }


    template<class T>
    void upload(MatrixAdapter<T> hostMat,
                MatrixAdapter<T>& preAlloc)
    {
        // not const ref because sigh...

        assert(hostMat.rows==preAlloc.rows);
        assert(hostMat.cols==preAlloc.cols);
        assert(preAlloc.stride==preAlloc.stride);



        char* a=(char*)hostMat.getData();
        char* b=(char*)preAlloc.getData();
        copy<char>(a, b,hostMat.dataSize(),pool.stream(0));

    }

    template<class T>
    MatrixAdapter<T> upload(const MatrixAdapter<T>& M){
        MatrixAdapter<T> out=allocate<T>(M.rows,M.cols);
        upload(M,out);
        return out;
    }






    template<class T>
    // does not take ownership
    void download(MatrixAdapter<T> devMatrix, MatrixAdapter<T>& preAlloc){
        assert(devMatrix.rows==preAlloc.rows);
        assert(devMatrix.cols==preAlloc.cols);
        assert(devMatrix.stride==preAlloc.stride);
        assert(devMatrix.getData()!=nullptr);
        assert(preAlloc.getData()!=nullptr);

        char* a=(char*)devMatrix.getData();
        char* b=(char*)preAlloc.getData();
        copy<char>(a, b, devMatrix.dataSize(), pool.stream(0));

    }
    template<class T>
    // does not take ownership
    MatrixAdapter<T> download(const MatrixAdapter<T>& devMatrix){
        T* data=(T*)(new char[devMatrix.rows*devMatrix.stride]);
        MatrixAdapter<T> ret(data,devMatrix.rows,devMatrix.cols,devMatrix.stride);
        download(devMatrix, ret);
        return ret;
    }


    template<class T>
    MatrixAdapter<T> upload(const std::vector<T>& v){
        auto m=MatrixAdapter<T>::allocate(1,v.size());
        for(int col=0;col<m.cols;++col)
            m(0,col)=v[col];

        auto devm= upload(m);
        m.release();
        return devm;
    }
    template<class T>
    void  upload(const std::vector<T>& v,MatrixAdapter<T>& devm){
        auto m=MatrixAdapter<T>::allocate(1,v.size()); // host mat
        for(int col=0;col<m.cols;++col)
            m(0,col)=v[col];

        upload(m,devm);
        m.release();
    }

    template<class T>
    std::vector<T> download2vector(MatrixAdapter<T> devMatrix){
        auto m=download(devMatrix);
        std::vector<T> v;v.reserve(m.rows*m.cols);
        for(uint row=0;row<devMatrix.rows;++row)
            for(uint col=0;col<devMatrix.cols;++col)
                v.push_back(m(row,col));
        m.release();
        return v;
    }


    void synchronize();
    DevStreamPool pool;
    // should never be copied!
    DevMemManager(DevMemManager const&) = delete;
    DevMemManager& operator=(DevMemManager const&) = delete;

private:

    std::mutex mtx;
    std::vector<void*> allocs;
    int next=0;
};





}// end namespace cvl
