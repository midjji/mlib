#pragma once
#include <cassert>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <mlib/utils/mlog/log.h>
#include <vector>



#define mhere() {\
    cudaDeviceSynchronize();\
    auto err=cudaGetLastError();\
    if(err!=cudaSuccess) {\
    mlog()<<cudaGetErrorName(err)<<": "<<cudaGetErrorString(err)<<"\n"; exit(1);}\
    }


//#define mhere()

template<class T>
class Tex2
{
public:
    T* data;
    int rows, cols, stride;
    __host__ __device__ Tex2(){}
    __host__ __device__ Tex2(T* data, int rows, int cols, int stride):data(data),rows(rows),cols(cols),stride(stride){}
    __host__ __device__ inline T& operator()(int row, int col){return data[row*stride +col];}
    __host__ __device__ inline const T& operator()(int row, int col) const{return data[row*stride +col];}
    __host__ __device__
    inline T* begin() {        return data;    }
    __host__ __device__
    inline T* end() {return &data[rows*stride];}
    __host__ __device__
    inline const T* begin() const{        return data;    }
    __host__ __device__
    inline const T* end() const{return &data[rows*stride];}
};


template< class T, bool device>
/**
 * @brief The Texture class
 *
 * Uses explicit striding which matches gpu texture requirements,
 * also uses pinned host memory and uses the A=B for transfer between host and device and vice versa.
 * Reserves and reuses memory if possible.
 */
class Texture
{
    //------------------------------------------------------------------
public:
    Texture()=default;

    // cant copy construct these
    Texture(const Texture&) =delete;



    // owns its pointer,
    ~Texture(){        free();    }



    T* data() { return data_; }
    const T* cdata() const { return data_; }
    int cols() const {return cols_;}
    int rows() const {return rows_;}
    int elements() const {return rows_*cols_;}
    // texture like object with worse performance? mtex?


    int stride() const{return stride_;}
    int bytes() const {return stride_*rows_*sizeof(T);}
    std::string str() const {
        std::stringstream ss;
        ss<<"PCIM: "<<rows()<<" "<<cols()<<" "<<stride()<<", cap="<<capacity_<<", "<<device<<": "<<data_;
        return ss.str();
    }
    bool same_size_and_stride(const Texture& im) const {
        if(im.cols()!=cols()) return false;
        if(im.rows()!=rows()) return false;
        if(im.stride()!=stride()) return false;
        return true;
    }

    template<class Image> void set_to_image(const Image& image)
    {
        // note be explicit here, since opencv stride is different
        resize_rc(image.rows,image.cols);
        for(int r=0;r<image.rows;++r)
            for(int c=0;c<image.cols;++c)
                at(r,c)=image(r,c);
    }
    void set_to_vec(const std::vector<T>& es)
    {
        // note be explicit here, since opencv stride is different
        resize_rc(es.size(),1);
        for(int r=0;r<es.size();++r)
                at(r,0)=es[r];
    }



private:
    static T* allocate_impl(int elements)
    {
        //mlog()<<"elements*sizeof(T): "<<elements*sizeof(T)<<"\n";
        T* data=nullptr;
        cudaError_t error;
        // gives aligned memory.
        if(device)
            error = cudaMalloc((void **) &data, elements*sizeof(T));
        else
            error= cudaHostAlloc	(	(void **) &data, elements*sizeof(T), cudaHostAllocPortable);

        if ((error != cudaSuccess)|| (data==nullptr))
        {
            std::cout<<"Failed to allocate memory on: "<<elements<<" size is: "<<sizeof(T)*elements/(1024.0*1024)<<"MB"<< "cuda error code is "<<(int)error<<" which means "<<cudaGetErrorString(error)<<std::endl;
            return nullptr;
        }
        return data;
    }
    void free()
    {
        //mlog()<<"freeing texture\n";
        if(data_==nullptr)
        {
            cols_ = 0;
            rows_ = 0;
            stride_=0;
            capacity_=0;
            return;
        }

        if(device)
            cudaFree(data_);
        else
            cudaFreeHost(data_);
        data_ = nullptr;
        cols_ = 0;
        rows_ = 0;
        stride_=0;
        capacity_=0;
    }
    void reserve(int elements)
    {
        if(elements<1024) elements=1024;
        if(elements<=capacity_) return;


        free();
        // either on device or page locked
        data_ = allocate_impl(elements);
        capacity_=elements;
    }
    static int stride_of(int rows, int cols)
    {
        if(rows==0) return 0;
        if(cols==0) return 0;
        if(rows==1) return cols;
        if(cols==1) return cols;
        return ((cols+31)/32)*32;
    }
    bool resize(int rows, int cols, int stride )
    {

        // if its already the same size, do nothing.
        // this is more of a reserve than a resize!

        if(stride<cols)
        {
            std::cout<<"wtf?!<<stride<cols<<"<<stride<<" "<<cols<<" "<<rows<<std::endl;

        }
        if(cols==0||rows==0)
        {
            std::cout<<"resize to empty?"<<std::endl;

        }

        reserve(stride*rows);
        stride_=stride;
        cols_=cols;
        rows_=rows;

        return true;
    }
public:
    void resize_rcs(int rows, int cols, int stride){
        resize(rows,cols,stride);
    }
    void resize_rc(int rows, int cols){
        resize_rcs(rows,cols,stride_of(rows, cols));
    }
    void resize_array(int elements){
        resize_rcs(1, elements,elements);
    }
    template<class A,  bool b >
    void resize( const Texture<A,b>& t)
    {
        resize_rcs(  t.rows(), t.cols(), t.stride());
    }

    // Set the image to a given byte value
    void set( T t ){
        if(data()==nullptr) return;

        set2(*this, t);
    }

    // use row=0, col instead for now
    //T& operator()(int index)    {        return this->operator()(index % stride_, index/stride_);    }
    inline T& operator()(int row, int col)
    {
        if (device)            require(!device," not available...");
        return data_[row*stride_ +col];
    }
    inline T& at(int row, int col)
    {
        if (device)
            require(!device," not available...");

        return data_[row*stride_ +col];
    }
    inline const T operator()(int row, int col) const{
        if (device)
            require(!device," not available...");
        return data_[row*stride_ +col];
    }
    cudaTextureObject_t texture() const
    {

        // create texture object
        cudaResourceDesc rd;
        memset(&rd, 0, sizeof(rd));
        /* //possible advantage for arrays... not important, they are all fast...
         rd.resType = cudaResourceTypeLinear; // do linear interpolation
        rd.res.linear.devPtr = buffer;
        rd.res.linear.desc.f = cudaChannelFormatKindFloat;
        rd.res.linear.desc.x = 32; // bits per channel
        rd.res.linear.sizeInBytes = N*sizeof(float);
        */
        rd.resType=cudaResourceType::cudaResourceTypePitch2D;
        rd.res.pitch2D.desc=cudaCreateChannelDesc<T>();
        rd.res.pitch2D.devPtr=data_;
        rd.res.pitch2D.height=rows();
        rd.res.pitch2D.width=cols();
        rd.res.pitch2D.pitchInBytes=stride()*sizeof(T);

        cudaTextureDesc td;
        memset(&td, 0, sizeof(td)); // default it

        td.addressMode[0]=cudaTextureAddressMode::cudaAddressModeClamp;
        td.addressMode[1]=cudaTextureAddressMode::cudaAddressModeClamp;
        td.addressMode[2]=cudaTextureAddressMode::cudaAddressModeClamp;


        td.filterMode=cudaTextureFilterMode::cudaFilterModeLinear;
        td.readMode = cudaReadModeElementType;

        // create texture object: we only have to do this once!
        cudaTextureObject_t tex=0; // its just a int, but thats too hard to return... fuck C
        // last nullptr is for view, i.e. submatrix?

        auto worked=[](cudaError_t error, std::string msg){
            if(error==cudaError::cudaSuccess) return; // no error
            std::cout<<msg<<": "<<cudaGetErrorString(error)<<" :"<<error<<std::endl;
        };
        worked(cudaCreateTextureObject(&tex, &rd, &td, nullptr),"failed to make texture?" );
        return tex;
    }
    Tex2<T> tex2(){return Tex2<T>(data_, rows_, cols_, stride_);}


    template<bool dev>
    Texture< T, device >& operator=( const Texture< T, dev >& from )
    {
        if(from.cdata()==cdata()) return *this;
        if(from.cdata()==nullptr){
            mlog()<<"forgot to initialize source\n";
            exit(1);
        }
        // reuses the old buffert if possible

        resize(from);


        auto worked=[](cudaError_t error, std::string msg){
            if(error==cudaError::cudaSuccess) return; // no error
            std::cout<<msg<<": "<<cudaGetErrorString(error)<<" :"<<error<<std::endl;
        };

        if(!device && !dev)
            worked(cudaMemcpy(data_, from.cdata(), from.bytes(), cudaMemcpyHostToHost), "copy  assign from h2h:");
        if(!dev && device )
            worked(cudaMemcpy(data_, from.cdata(), from.bytes(), cudaMemcpyHostToDevice), "copy  assign from h2d:");
        if(dev && !device )
            worked(cudaMemcpy(data_, from.cdata(), from.bytes(), cudaMemcpyDeviceToHost), "copy  assign from d2h:");
        if(device && dev)
            worked(cudaMemcpy(data_, from.cdata(), from.bytes(), cudaMemcpyDeviceToDevice),"copy  assign from d2d: ");

        // host, dev, is inferred from pointers, does not seem to work?
        //cuda_worked<0>(cudaMemcpy(data_, from.data(), from.bytes(), cudaMemcpyDefault),"copy assign");
        return *this;
    }

private:
    int cols_=0;
    int rows_=0;
    /// image strides in elements, not bytes
    int stride_=0;
    int capacity_=0;
    T* data_=nullptr;
};




